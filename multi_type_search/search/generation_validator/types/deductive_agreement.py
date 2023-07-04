from multi_type_search.search.generation_validator import GenerationValidator
from multi_type_search.search.graph import Node, HyperNode, Graph, GraphKeyTypes, compose_index
from multi_type_search.search.step_selector import Step
from multi_type_search.search.step_type import StepType
from multi_type_search.search.comparison_metric import ComparisonMetric

from typing import List, Tuple, Dict
import torch


class DeductiveAgreementValidator(GenerationValidator):
    """
    Given an Abductive HyperNode we will try to validate it by regenerating the conclusion statement (the last argument
    passed into the abductive steptype) given the initial arguments of the Abductive Hypernode and the generations it
    created.  See the validate() method for an example.
    """

    search_obj_type: str = 'deductive_agreement_generation_validator'

    deductive_steptype: StepType
    comparison_metric: ComparisonMetric
    threshold: float

    def __init__(
            self,
            *args,
            deductive_steptype: StepType,
            comparison_metric: ComparisonMetric,
            threshold: float,
            **kwargs
    ):
        """
        :param args:
        :param deductive_steptype: A DeductiveStepType SearchObject to recreate the conclusion argument
        :param comparison_metric: The comparison metric for checking if the generations from the deductive model
            recovered the conclusion of the abduction.
        :param threshold: The threshold on the comparison_metric where if the score is at or above it's value, then the
            generation has successfully recovered the conclusion.
        :param kwargs:
        """

        super().__init__(*args, **kwargs)
        self.deductive_steptype = deductive_steptype
        self.comparison_metric = comparison_metric
        self.threshold = threshold

    def validate(
            self,
            graph: Graph,
            step: Step,
            new_premises: List[Node] = (),
            new_abductions: List[HyperNode] = (),
            new_deductions: List[HyperNode] = ()
    ) -> Tuple[List[Node], List[HyperNode], List[HyperNode]]:
        """
        Given an abductive hypernode, take the first N - 1 arguments as well as each individual generation in the
        HyperNode and attempt to recreate the last argument (the conclusion statement for all abductive steps). For
        example:

        Abductive(P0, G) -> A

        Then we will attempt to validate the generation A via

        Deductive(P0, A) -> ~G

        Then we will attempt measure if G~ recovered G making the abductive generation A valid.

        COMPARE(~G, G) > Threshold

        If the compare metric is above the set threshold, then the abductive generation A is considered valid.

        :param graph: The graph where all the arguments can be looked up
        :param step: The step that was used to generate the abduction
        :param new_premises: New premises (ignored here)
        :param new_abductions: New Abductions we want to validate
        :param new_deductions: New Deductions (ignored here)
        :return: New Premises, newly validated abductions, and new deductions as a tuple.
        """

        if len(new_abductions) == 0:
            return new_premises, new_abductions, new_deductions

        # Get all the arguments of the abductive model except the last one and combine them with each individual
        # generation (Node) to produce prompts we want to use in the deductive model to recover the last argument.
        generation_arguments = [
            self.deductive_steptype.format_stepmodel_input([
                *[graph[x].normalized_value for x in hypernode.arguments[0:-1]], node.normalized_value
            ]) for hypernode in new_abductions for node in hypernode.nodes
        ]

        # The conclusive statement is always the last argument of the abductive model thus the target
        targets = [
            graph[hypernode.arguments[-1]].normalized_value for hypernode in new_abductions for _ in hypernode.nodes
        ]

        # So we can keep track of which prompt/target pair belongs to which hypernode.
        hypernode_map = [
            (hypernode, node) for hypernode in new_abductions for node in hypernode.nodes
        ]

        # Produce the Deductive generations that are attempting to recover the conclusive statement in the Abductive
        # HyperNode.
        deductive_generations = [
            x[0] for x in self.deductive_steptype.step_model.sample(text=generation_arguments, sample_override=1)
        ]

        # Score each generated conclusion with the actual conclusion
        scores = self.comparison_metric.score(deductive_generations, targets)

        # For every node, check whether the deductive step model recovered it's conclusive statement.
        for (hypernode, node), score in zip(hypernode_map, scores):
            if score < self.threshold:
                hypernode.nodes.remove(node)

        # Remove all hypernodes that had all of it's nodes removed (because they were all invalid)
        validated_new_abductions = []
        for hypernode in new_abductions:
            if len(hypernode.nodes) == 0:
                continue
            validated_new_abductions.append(hypernode)

        return new_premises, validated_new_abductions, new_deductions

    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        return self.search_obj_type, {
            'deductive_steptype': self.deductive_steptype.to_json_config(),
            'comparison_metric': self.comparison_metric,
            'threshold': self.threshold,
        }

    def to(self, device: str) -> 'DeductiveAgreementValidator':
        deductive_steptype = self.deductive_steptype.to(device)
        comparison_metric = self.comparison_metric.to(device)

        return DeductiveAgreementValidator(
            deductive_steptype=deductive_steptype,
            comparison_metric=comparison_metric,
            threshold=self.threshold,
        )
