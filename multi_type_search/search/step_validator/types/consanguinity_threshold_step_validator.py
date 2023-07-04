from typing import List, Tuple, Union, Dict, Optional
import itertools

from multi_type_search.search.step_validator import StepValidator
from multi_type_search.search.graph import Graph, GraphKeyTypes, compose_index, decompose_index
from multi_type_search.search.step_selector import Step


class ConsanguinityThresholdStepValidator(StepValidator):
    """
    This Step Validator enforces that the arguments used to create a step are unique up to some level of depth in the
    graph.
    ---
    Example:

    P0 + P0 -> D1 (this step has a consanguinity of 1 because the inputs are exactly the same for the current step.)

    Another Example:

    P0 + P1 -> D0
    P0 + P1 -> D1

    D0 + D1 -> D2 (this step has a consanguinity of 2 because the inputs, i0 and i1, are made of the same ancestors)
    """

    search_obj_type: str = 'consanguinity_threshold_step_validator'

    threshold: int
    compare_subindex: bool

    def __init__(
            self,
            *args,
            threshold: int = None,
            compare_subindex: bool = True,
            **kwargs
    ):
        """
        :param args:
        :param threshold: How far up the graph before ancestor arguments can mix.  (1 means siblings cannot match, 2
            means parents cannot overlap, 3 means grandparents, etc. etc.)
        :param compare_subindex: When finding ancestors, use the subindex to further distinguish arguments (if this is
            false, ancestry will be determined by shared hypernodes)
        :param kwargs:
        """

        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.compare_subindex = compare_subindex

    def get_all_ancestor_inputs(self, graph: Graph, index: str, depth: int, current_depth: int = 0) -> List[str]:
        """
        Returns a list (including duplicates) of ancestors of some step.

        :param graph: The graph where these steps live
        :param index: The graph index of the step we want to find ancestors for
        :param depth: The depth of which we want to go into the graph from the step
        :param current_depth: The current depth of the search for ancestors
        :return: List of all the steps/ancestors required to make the current step
        """

        # Either we've met the depth specified or we want to ignore the goal node.
        # We ignore the goal node for abductive steps which will often generate from the goal node at some level and all
        # must come from the goal node (any value of the threshold would filter out abductive steps).
        step_key, imain_idx, isub_idx = decompose_index(index)
        if current_depth >= depth or step_key == GraphKeyTypes.GOAL:
            return []

        ancestor = compose_index(step_key, imain_idx, isub_idx) if self.compare_subindex is True else compose_index(step_key, imain_idx)

        hypernode = graph.get_hypernode(index)
        if hypernode is None:
            return [ancestor]

        return [ancestor, *list(itertools.chain(*[
            self.get_all_ancestor_inputs(graph, x, depth, current_depth=current_depth+1)
            for x in hypernode.arguments
        ]))]

    def check_step(self, graph: Graph, step: Step, threshold: int) -> Optional[Step]:
        """
        Given a step, check to see if it's ancestors overlap within some threshold (distance)

        :param graph: The graph where all the arguments (ancestors) live
        :param step: The current step we want to validate
        :param threshold: The depth of ancestors to check against
        :return: Either the step or None (none means the step was invalid)
        """

        if len(step.arguments) < 2:
            return step

        ancestors = [
            set([
                ancestor for ancestor in
                self.get_all_ancestor_inputs(graph, x, depth=threshold)
                if len(ancestor) > 0
            ])
            for x in step.arguments
        ]

        for idx, ancestor in enumerate(ancestors[0:-1]):
            if sum([len(ancestor.intersection(x)) for x in ancestors[idx + 1:]]) > 0:
                return None

        return step

    def validate(
            self,
            graph: Graph,
            unvalidated_steps: List[Step] = (),
    ) -> List[Step]:
        """
        Given new potential steps make sure the inputs used to create those steps have independent sets of
        ancestors up to some depth.

        :param graph: A graph with the nodes of the arguments in each step
        :param unvalidated_steps: Unvalidated steps
        :return: Validated steps
        """

        valid_steps = []

        for step in unvalidated_steps:
            valid_step = self.check_step(graph, step, self.threshold)
            if valid_step:
                valid_steps.append(valid_step)

        return valid_steps

    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        return self.search_obj_type, {
            'threshold': self.threshold
        }
