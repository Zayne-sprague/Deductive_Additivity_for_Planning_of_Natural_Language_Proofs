from multi_type_search.search.generation_validator import GenerationValidator
from multi_type_search.search.graph import Node, HyperNode, Graph
from multi_type_search.search.step_type import StepType
from multi_type_search.search.step_selector import Step
from multi_type_search.search.comparison_metric import ComparisonMetric

from typing import List, Tuple, Dict
from copy import deepcopy


class AbductiveAgreementValidator(GenerationValidator):
    """
    Given a Deductive generation, this validator will attempt to recreate the arguments used in the deduction.  If the
    Abductive model is able to recreate the arguments, then the deduction is considered valid.
    """

    search_obj_type: str = 'abductive_agreement_generation_validator'

    abductive_steptype: StepType
    comparison_metric: ComparisonMetric
    threshold: float
    invalid_argument_tolerance: int

    def __init__(
            self,
            *args,
            abductive_steptype: StepType,
            comparison_metric: ComparisonMetric,
            threshold: float,
            invalid_argument_tolerance: int = 0,
            **kwargs
    ):
        """
        :param args:
        :param abductive_steptype: An AbductiveStepType SearchObject to recreate the deductive arguments
        :param comparison_metric: The comparison metric for checking if the generations from the abductive model
            recovered the arguments of the deduction.
        :param threshold: The threshold on the comparison_metric where if the score is at or above it's value, then the
            generation has successfully recovered the argument.
        :param invalid_argument_tolerance: Most Deductive Steps will have 2 arguments, this controls the number of
            arguments that are allowed to NOT be recovered.  I.E. at val 1, only 1 of the 2 args have to be regenerated
            by the abductive model.
        :param kwargs:
        """

        super().__init__(*args, **kwargs)
        self.abductive_steptype = abductive_steptype
        self.comparison_metric = comparison_metric
        self.threshold = threshold
        self.invalid_argument_tolerance = invalid_argument_tolerance

    def validate(
            self,
            graph: Graph,
            step: Step,
            new_premises: List[Node] = (),
            new_abductions: List[HyperNode] = (),
            new_deductions: List[HyperNode] = ()
    ) -> Tuple[List[Node], List[HyperNode], List[HyperNode]]:
        """
        This function is meant to validate new deductions by trying to recover the individual arguments used to create
        the new deduction via the abductive model.  In other words, if a new deduction is of the form

        P1 + P2 = D

        We validate this by doing

        D - P1 = ~P2
        D - P2 = ~P1

        To check if ~P2 and ~P1 recovered the origianl P2 and P1 we run a comparison metric on them and threshold it

        COMPARE(~P1, P1) > Threshold
        COMPARE(~P2, P2) > Threshold

        Because we do this for multiple arguments we check to see if the number of valid argument regenerations is above
        the invalid_argument_threshold

        VALID_ARGUMENT_REGENERATIONS > invalid_argument_threshold

        If a hypernode's generation has enough valid regenerated arguments, then the generation is considered valid.

        :param graph: The current graph that has the arguments for the deductions that can be looked up
        :param step: The step we took to generate the new deductions
        :param new_premises: New premises (they'll be ignored here)
        :param new_abductions: New abductions (they'll be ignored here)
        :param new_deductions: New deductions to validate
        :return: All the new premises, new abductions, and the newly validated deductions as a tuple.
        """

        if len(new_deductions) == 0:
            return new_premises, new_abductions, new_deductions

        # Pretty complex statement to just get a combination of inputs of the form
        # " ALL ARGUMENTS EXCEPT 1 " + " OUTPUT OF THE NODE " which we will use to get the 1 argument that was left out.
        generation_arguments = [
            self.abductive_steptype.format_stepmodel_input([
                *[graph[x].normalized_value for x in [a for idx, a in enumerate(hypernode.arguments) if idx != aidx]], node.normalized_value
            ])
            for hypernode in new_deductions for node in hypernode.nodes for aidx, arg in enumerate(hypernode.arguments)
        ]

        # Set the targets we want to regenerate through the abductive model (always will be the argument we left out in
        # the generation_arguments declaration).
        targets = [
            graph[arg].normalized_value
            for hypernode in new_deductions for _ in hypernode.nodes for arg in hypernode.arguments
        ]

        # So we can easily keep track of which generation/target pair links to which hypernode/deduction
        hypernode_map = [
            (hypernode, node)
            for hypernode in new_deductions for node in hypernode.nodes for _ in hypernode.arguments
        ]

        # Get the generations from the abductive model
        abductive_generations = [
            x[0] for x in self.abductive_steptype.step_model.sample(generation_arguments, sample_override=1)
        ]

        # Score each abductive generation and actual argument combo to see how well we recovered the arguments.
        scores = self.comparison_metric.score(abductive_generations, targets)

        # For every Deductive HyperNode, check to see if we regenerated the arguments
        for (hypernode, node), score in zip(hypernode_map, scores):
            if score < self.threshold and node in hypernode.nodes:
                invalid_count = node.tags.get('abductive_agreement_invalid_args', 0)
                invalid_count += 1
                node.tags['abductive_agreement_invalid_args'] = invalid_count

                # Depending on how many invalid arguments we all, either remove the node from the list of valid
                # nodes in the deductive hypernode or continue on.
                if invalid_count > self.invalid_argument_tolerance:
                    hypernode.nodes.remove(node)

        validated_new_deductions = []
        for hypernode in new_deductions:

            # If any of the deductive hypernodes had all their nodes removed, do not add it to the list of valid
            # deductions.
            if len(hypernode.nodes) == 0:
                continue
            validated_new_deductions.append(hypernode)

        return new_premises, new_abductions, validated_new_deductions

    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        return self.search_obj_type, {
            'abductive_steptype': self.abductive_steptype.to_json_config(),
            'comparison_metric': self.comparison_metric,
            'threshold': self.threshold,
            'invalid_argument_tolerance': self.invalid_argument_tolerance
        }

    def to(self, device: str) -> 'AbductiveAgreementValidator':
        abductive_steptype = self.abductive_steptype.to(device)
        comparison_metric = self.comparison_metric.to(device)

        return AbductiveAgreementValidator(
            abductive_steptype=abductive_steptype,
            comparison_metric=comparison_metric,
            threshold=self.threshold,
            invalid_argument_tolerance=self.invalid_argument_tolerance
        )
