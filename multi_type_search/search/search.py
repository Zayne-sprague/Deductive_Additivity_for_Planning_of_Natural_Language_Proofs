from multi_type_search.search.graph import Node, HyperNode, Graph
from multi_type_search.search.step_type import StepType
from multi_type_search.search.step_selector import StepSelector, Step
from multi_type_search.search.generation_validator import GenerationValidator
from multi_type_search.search.step_validator import StepValidator
from multi_type_search.search.termination_criteria import TerminationCriteria
from multi_type_search.search.premise_retriever import PremiseRetriever

from typing import List, Dict, Callable, Tuple


class Search:
    registered_callbacks: Dict[str, List[Callable]]

    def __init__(
            self,
    ):
        """
        Base search class that will handle the over-arching search and fringe.  It can use multiple step type models
        and prioritize their individual steps executing one step at a time.
        """

        self.registered_callbacks = {}

    def search(
            self,
            graph: Graph,
            step_selector: StepSelector,
            step_types: List[StepType],
            max_steps: int,
            step_validators: List[StepValidator] = (),
            generation_validators: List[GenerationValidator] = (),
            termination_criteria: List[TerminationCriteria] = (),
            premise_retriever: PremiseRetriever = None
    ) -> Graph:
        """
        The actual search.

        Given a list of premises, a hypothesis, and a goal state you want to reach (could be the hypothesis a second
        time) -- iterate through the step types to create new prioritized steps and execute them one at a time until
        either a termination criterion is met or we hit the maximum number of steps.

        The heuristic will rank the individual steps.

        :param graph: The graph to search over.
        :param step_selector: StepSelector for managing which step to take next
        :param step_types: The allowed step types the search can take
        :param max_steps: The maximum number of steps to perform in the search
        :param step_validators: Step validators to validate potential steps before added to the fringe.
        :param generation_validators: Generation validators to validate generations before they are added to the fringe.
        :param termination_criteria: A list of TerminationCriteria classes that can terminate the search early.
        :param premise_retriever: Reduces the set of premises on the initial graph to a subset (only run once)
        :return: An expanded Tree object with intermediates and hypotheses created during the inner loop of the search
            (does not always mean the search found the goal)
        """

        self.handle_start('main', graph, step_selector, step_types, max_steps, step_validators, generation_validators)

        step_selector = self.initial_population(graph, step_types, step_selector, step_validators, termination_criteria, premise_retriever)

        terminate = False

        # Inner loop of the search, sample the top most item from the fringes priority queue.
        for step_idx, steps in enumerate(step_selector):
            self.handle_start('step', step_idx, max_steps, steps, graph, step_selector)
            for step in steps:
                self.handle_start('generation_step', step_idx, max_steps, step, graph, step_selector)

                new_premises, new_abductions, new_deductions = self.sample_generations(step, graph)
                new_premises, new_abductions, new_deductions = self.validate_generations(
                    new_premises,
                    new_abductions,
                    new_deductions,
                    graph,
                    generation_validators,
                    step
                )

                # Generations cannot be added until we create the steps (this helps differentiate between current
                # Generations/Nodes and new Generations/Nodes
                new_steps = self.create_steps(graph, step_types, new_premises, new_abductions, new_deductions)
                self.add_generations(new_premises, new_abductions, new_deductions, graph)

                # Generations need to be added before we verify steps
                new_steps = self.validate_steps(new_steps, graph, step_validators)

                step_selector = self.add_steps(new_steps, graph, step_selector)

                self.handle_end('generation_step', step_idx, max_steps, step, graph, step_selector, new_premises, new_abductions, new_deductions)

                if any([x.should_terminate(new_premises, new_abductions, new_deductions, new_steps, graph, step_selector) for x in termination_criteria]):
                    terminate = True
                    break

            self.handle_end('step', step_idx, max_steps, steps, graph, step_selector)

            if step_idx >= max_steps - 1 or terminate:
                break

        self.handle_end('main', graph, step_selector, step_types, max_steps, step_validators, generation_validators)

        return graph

    def initial_population(
            self,
            graph: Graph,
            step_types: List[StepType],
            step_selector: StepSelector,
            step_validators: List[StepValidator],
            termination_criteria: List[TerminationCriteria],
            premise_retriever: PremiseRetriever = None
    ) -> StepSelector:
        self.handle_start('initial_population', graph, step_types, step_selector)

        [x.reset() for x in termination_criteria]
        step_selector.reset()

        if premise_retriever:
            graph.premises = premise_retriever.reduce(graph.premises, graph.goal)
            
        steps = self.create_steps(Graph(graph.goal), step_types, graph.premises, graph.abductions, graph.deductions)
        steps = self.validate_steps(steps, graph, step_validators)
        step_selector = self.add_steps(steps, graph, step_selector)

        self.handle_end('initial_population', steps, graph, step_types, step_selector)
        return step_selector

    def create_steps(
            self,
            graph: Graph,
            step_types: List[StepType],
            new_premises: List[Node],
            new_abductions: List[HyperNode],
            new_deductions: List[HyperNode]
    ) -> List[Step]:
        self.handle_start('create_steps', graph, step_types, new_premises, new_abductions, new_deductions)
        steps = []
        for step_type in step_types:
            steps.extend(
                step_type.generate_step_combinations(
                    graph,
                    new_premises,
                    new_abductions,
                    new_deductions
                )
            )
        self.handle_end('create_steps', steps, graph, step_types, new_premises, new_abductions, new_deductions)
        return steps

    def validate_steps(
            self,
            steps: List[Step],
            graph: Graph,
            step_validators: List[StepValidator]
    ):
        if len(step_validators) == 0:
            return steps

        self.handle_start('validate_steps', steps, graph, step_validators)

        for validator in step_validators:
            steps = validator.validate(graph, steps)

        self.handle_end('validate_steps', steps, graph, step_validators)
        return steps

    def add_steps(
            self,
            steps: List[Step],
            graph: Graph,
            step_selector: StepSelector
    ):
        self.handle_start('add_steps', steps, graph, step_selector)
        step_selector.add_steps(steps, graph)
        self.handle_end('add_steps', steps, graph, step_selector)
        return step_selector

    def sample_generations(
            self,
            step: Step,
            graph: Graph
    ) -> Tuple[List[Node], List[HyperNode], List[HyperNode]]:
        self.handle_start('sample_generations', step, graph)

        args = step.arguments
        step_type: StepType = step.type
        step_model = step_type.step_model

        # Get step generations from the step model.
        formatted_input = step_type.format_stepmodel_input([graph[x].normalized_value for x in args])

        step_generations = step_model.sample(formatted_input)

        # Each step type generates a different output (deductive generates deductions, abductive generate
        # abductions for example) - instead of a big if statement, we allow the class to return the new items.
        new_premises, new_abductions, new_deductions = step_type.build_hypernodes(
            step_generations,
            step
        )

        self.handle_end('sample_generations', new_premises, new_abductions, new_deductions, step, graph)
        return new_premises, new_abductions, new_deductions

    def validate_generations(
            self,
            new_premises: List[Node],
            new_abductions: List[HyperNode],
            new_deductions: List[HyperNode],
            graph: Graph,
            generation_validators: List[GenerationValidator],
            step: Step

    ):
        if len(generation_validators) == 0:
            return new_premises, new_abductions, new_deductions

        self.handle_start('validate_generations', new_premises, new_abductions, new_deductions, graph, generation_validators, step)

        # Check with each validator and make sure the new generated statements are valid.
        for validator in generation_validators:
            new_premises, new_abductions, new_deductions = validator.validate(
                graph=graph,
                step=step,
                new_premises=new_premises,
                new_abductions=new_abductions,
                new_deductions=new_deductions
            )

        self.handle_end('validate_generations', new_premises, new_abductions, new_deductions, graph, generation_validators, step)
        return new_premises, new_abductions, new_deductions

    def add_generations(
            self,
            new_premises: List[Node],
            new_abductions: List[HyperNode],
            new_deductions: List[HyperNode],
            graph: Graph,
    ):

        self.handle_start('add_generations', new_premises, new_abductions, new_deductions, graph)
        graph.premises = [*graph.premises, *new_premises]
        graph.abductions = [*graph.abductions, *new_abductions]
        graph.deductions = [*graph.deductions, *new_deductions]
        self.handle_end('add_generations', new_premises, new_abductions, new_deductions, graph)

    def register_hook(self, lifecycle: str, hook: Callable):
        hooks = self.registered_callbacks.get(lifecycle, [])
        hooks.append(hook)
        self.registered_callbacks[lifecycle] = hooks

    def handle_start(self, lifecycle: str, *args, **kwargs):
        hooks = self.registered_callbacks.get(f'{lifecycle}_start', [])
        for hook in hooks:
            hook(*args, **kwargs)

    def handle_end(self, lifecycle: str, *args, **kwargs):
        hooks = self.registered_callbacks.get(f'{lifecycle}_end', [])
        for hook in hooks:
            hook(*args, **kwargs)


