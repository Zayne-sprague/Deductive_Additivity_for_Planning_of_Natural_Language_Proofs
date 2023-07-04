from multi_type_search.search.generation_validator import GenerationValidator
from multi_type_search.search.graph import Node, HyperNode, Graph, GraphKeyTypes, compose_index
from multi_type_search.search.step_selector import Step
from multi_type_search.search.comparison_metric import ComparisonMetric

from typing import List, Tuple, Dict


class DeduplicationValidator(GenerationValidator):
    """
    This generation validator attempts to remove any premises, abductions, or deductions that are already in the
    graph by measuring their similarity through a comparison metric which is thresholded.
    """

    search_obj_type: str = 'deduplication_generation_validator'

    comparison: ComparisonMetric
    threshold: float
    check_arguments: bool
    check_matching_generation_type: bool
    normalize_values: bool

    def __init__(
            self,
            *args,
            comparison: ComparisonMetric,
            threshold: float,
            check_arguments: bool = True,
            check_matching_generation_type: bool = True,
            check_other_generations: bool = True,
            normalize_values: bool = True,
            **kwargs
    ):
        """
        :param args: N/A
        :param comparison: The comparison metric to use to measure the similarity between generations.
        :param threshold: Threshold to set for the comparison metric (if greater than this value, the generation will
            be removed.)
        :param check_arguments: Check the arguments used to create the generation (regardless of the class of the
        argument)
        :param check_matching_generation_type: Check all the corresponding generations of the same type (if a premise
            was introduced, check for dupes in all other premises)
        :param normalize_values: Use the normalized value of the nodes instead of the raw values.
        :param kwargs: N/A
        """

        super().__init__(*args, **kwargs)
        self.comparison = comparison
        self.threshold = threshold
        self.check_arguments = check_arguments
        self.check_matching_generation_type = check_matching_generation_type
        self.check_other_generations = check_other_generations
        self.normalize_values = normalize_values

    def validate(
            self,
            graph: Graph,
            step: Step,
            new_premises: List[Node] = (),
            new_abductions: List[HyperNode] = (),
            new_deductions: List[HyperNode] = ()
    ) -> Tuple[List[Node], List[HyperNode], List[HyperNode]]:
        """
        :param graph: The current graph with all other generations/arguments
        :param step: The step used to generate the new data
        :param new_premises: New premises to validate
        :param new_abductions: New abductions to validate
        :param new_deductions: New deductions to validate
        :return: Newly validated premises, newly validated abductions, and newly validated deductions as a tuple
        """

        validated_new_premises = []
        validated_new_abductions = []
        validated_new_deductions = []

        for pidx, new_premise in enumerate(new_premises):
            targets = []
            if self.check_arguments:
                targets.extend(step.arguments)
            if self.check_matching_generation_type:
                targets.extend([compose_index(GraphKeyTypes.PREMISE, pidx) for pidx in range(len(graph.premises))])

            targets = [graph[x].normalized_value if self.normalize_values else x.value for x in targets]
            if self.check_other_generations:
                targets.extend([x.normalized_value if self.normalize_values else x.value for x in new_premises[pidx+1:]])
            group = [new_premise.normalized_value if self.normalize_values else new_premise.value] * len(targets)

            scores = self.comparison.score(group, targets)
            if any([x >= self.threshold for x in scores]):
                continue
            validated_new_premises.append(new_premise)

        for aidx, new_abduction in enumerate(new_abductions):
            validated_nodes = []
            for nix, node in enumerate(new_abduction.nodes):
                targets = []
                if self.check_arguments:
                    targets.extend(step.arguments)
                if self.check_matching_generation_type:
                    targets.extend([
                        compose_index(GraphKeyTypes.ABDUCTIVE, hidx, nidx)
                        for hidx, hypernode in enumerate(graph.abductions)
                        for nidx in range(len(hypernode.nodes))
                    ])


                targets = [graph[x].normalized_value if self.normalize_values else x.value for x in targets]
                if self.check_other_generations:
                    targets.extend([x.normalized_value if self.normalize_values else x.value for x in new_abduction.nodes[nidx+1:]])

                group = [node.normalized_value if self.normalize_values else node.value] * len(targets)

                scores = self.comparison.score(group, targets)
                if any([x >= self.threshold for x in scores]):
                    continue
                validated_nodes.append(node)

            if len(validated_nodes) == 0:
                continue

            new_abduction.nodes = validated_nodes
            validated_new_abductions.append(new_abduction)

        for didx, new_deduction in enumerate(new_deductions):
            validated_nodes = []
            for nidx, node in enumerate(new_deduction.nodes):
                targets = []
                if self.check_arguments:
                    targets.extend(step.arguments)
                if self.check_matching_generation_type:
                    targets.extend([
                        compose_index(GraphKeyTypes.DEDUCTIVE, hidx, nidx)
                        for hidx, hypernode in enumerate(graph.deductions)
                        for nidx in range(len(hypernode.nodes))
                    ])


                targets = [graph[x].normalized_value if self.normalize_values else x.value for x in targets]
                if self.check_other_generations:
                    targets.extend([x.normalized_value if self.normalize_values else node.value for x in new_deduction.nodes[nidx+1:]])

                group = [node.normalized_value if self.normalize_values else node.value] * len(targets)

                scores = self.comparison.score(group, targets)
                if any([x >= self.threshold for x in scores]):
                    continue
                validated_nodes.append(node)

            if len(validated_nodes) == 0:
                continue

            new_deduction.nodes = validated_nodes
            validated_new_deductions.append(new_deduction)

        return validated_new_premises, validated_new_abductions, validated_new_deductions

    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        return self.search_obj_type, {
            'comparison': self.comparison.to_json_config(),
            'threshold': self.threshold,
            'check_arguments': self.check_arguments,
            'check_matching_generation_type': self.check_matching_generation_type,
            'check_other_generations': self.check_other_generations,
            'normalize_values': self.normalize_values
        }

    def to(self, device: str) -> 'DeduplicationValidator':
        comparison = self.comparison.to(device)
        return DuplicateValidator(
            comparison=comparison,
            threshold=self.threshold,
            check_arguments=self.check_arguments,
            check_matching_generation_type=self.check_matching_generation_type,
            normalize_values=self.normalize_values
        )
