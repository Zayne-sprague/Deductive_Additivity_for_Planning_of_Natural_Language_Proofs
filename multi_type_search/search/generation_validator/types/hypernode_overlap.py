from multi_type_search.search.generation_validator import GenerationValidator
from multi_type_search.search.graph import Node, HyperNode, Graph, GraphKeyTypes, compose_index
from multi_type_search.search.step_selector import Step
from multi_type_search.search.comparison_metric import ComparisonMetric

from typing import List, Tuple, Dict


class HyperNodeOveralpValidator(GenerationValidator):
    """
    This generation validator measures the overlap between a new hypernode and an existing hypernode.  If this overlap
    (measured by a comparison metric + threshold) is above a percentage threshold, then the new hypernode is either
    merged into the existing hypernode or removed.

    TODO - This most likely should be it's own SearchObject type, one that maps generations to the correct HyperNode.
        This can be done later.
    """

    search_obj_type: str = 'hypernode_overlap_generation_validator'

    comparison: ComparisonMetric
    threshold: float
    overlap_percentage_threshold: float
    merge_hypernode: bool
    normalize_values: bool

    def __init__(
            self,
            *args,
            comparison: ComparisonMetric,
            threshold: float,
            overlap_percentage_threshold: float = 0.10,
            merge_hypernode: bool = False,
            normalize_values: bool = True,
            **kwargs
    ):
        """
        :param args: N/A
        :param comparison: The comparison metric to use to measure the similarity between generations.
        :param threshold: Threshold to set for the comparison metric (if greater than this value, the generation will
            be removed.)
        :param overlap_percentage_threshold: Percentage of overlap between the current hypernode and an existing
            hypernode
        :param merge_hypernode: Merge overlapping HyperNodes, otherwise, remove the new one.
        :param normalize_values: Use the normalized value of the nodes instead of the raw values.
        :param kwargs: N/A
        """

        super().__init__(*args, **kwargs)
        self.comparison = comparison
        self.threshold = threshold
        self.overlap_percentage_threshold = overlap_percentage_threshold
        self.merge_hypernode = merge_hypernode
        self.normalize_values = normalize_values

    def get_hypernode_overlap(
            self,
            hypernode: HyperNode,
            other_hypernode: HyperNode
    ) -> float:
        total = len(other_hypernode)

        group = [x.normalized_value if self.normalize_values else x.value for _ in other_hypernode.nodes for x in
                 hypernode.nodes]

        targets = [x.normalized_value if self.normalize_values else x.value for x in other_hypernode.nodes for _ in
                   hypernode.nodes]

        scores = self.comparison.score(group, targets)

        overlapped_gens = []
        for idx, score in enumerate(scores):
            other_hypernode_idx = idx % total
            if score > self.threshold and other_hypernode_idx not in overlapped_gens:
                overlapped_gens.append(other_hypernode_idx)

        percent_overlap = len(overlapped_gens) / total

        return percent_overlap

    def validate_hypernode(
            self,
            hypernode: HyperNode,
            other_hypernodes: List[HyperNode]
    ):
        for idx, other_hypernode in enumerate(other_hypernodes):
            overlap_percentage = self.get_hypernode_overlap(hypernode, other_hypernode)
            if overlap_percentage > self.overlap_percentage_threshold:
                return False, idx

        return True, -1

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

        validated_new_abductions = []
        validated_new_deductions = []

        for abduction in new_abductions:
            is_valid, overlapping_with = self.validate_hypernode(abduction, graph.abductions)

            if not is_valid:
                if self.merge_hypernode:
                    graph.abductions[overlapping_with].tags['hypernode_info'] = {
                        'graph_index': compose_index(GraphKeyTypes.ABDUCTIVE, overlapping_with),
                        'new_node_start_index': len(graph.abductions[overlapping_with])
                    }

                    graph.abductions[overlapping_with].nodes.extend(abduction.nodes)
                    abduction = graph.abductions[overlapping_with]
                else:
                    continue

            validated_new_abductions.append(abduction)

        for deduction in new_deductions:
            is_valid, overlapping_with = self.validate_hypernode(deduction, graph.deductions)

            if not is_valid:
                if self.merge_hypernode:
                    graph.deductions[overlapping_with].tags['hypernode_info'] = {
                        'graph_index': compose_index(GraphKeyTypes.DEDUCTIVE, overlapping_with),
                        'new_node_start_index': len(graph.deductions[overlapping_with])
                    }

                    graph.deductions[overlapping_with].nodes.extend(deduction.nodes)
                    deduction = graph.deductions[overlapping_with]
                else:
                    continue

            validated_new_deductions.append(deduction)

        return new_premises, validated_new_abductions, validated_new_deductions

    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        return self.search_obj_type, {
            'comparison': self.comparison.to_json_config(),
            'threshold': self.threshold,
            'overlap_percentage_threshold': self.overlap_percentage_threshold,
            'merge_hypernode': self.merge_hypernode,
            'normalize_values': self.normalize_values
        }

    def to(self, device: str) -> 'HyperNodeOveralpValidator':
        comparison = self.comparison.to(device)
        return DuplicateValidator(
            comparison=comparison,
            threshold=self.threshold,
            overlap_percentage_threshold=self.overlap_percentage_threshold,
            merge_hypernode=self.merge_hypernode,
            normalize_values=self.normalize_values
        )
