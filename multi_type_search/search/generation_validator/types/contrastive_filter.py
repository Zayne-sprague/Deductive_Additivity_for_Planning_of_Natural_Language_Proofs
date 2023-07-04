from multi_type_search.search.generation_validator import GenerationValidator
from multi_type_search.search.graph import Node, HyperNode, Graph
from multi_type_search.search.step_type import StepType
from multi_type_search.search.step_selector import Step
from multi_type_search.search.search_model import NodeEmbedder
from multi_type_search.search.comparison_metric import ComparisonMetric

from typing import List, Tuple, Dict
from copy import deepcopy
import torch


class ContrastiveFilterValidator(GenerationValidator):
    """

    """

    search_obj_type: str = 'contrastive_filter_validator'

    embedder: NodeEmbedder
    threshold: float
    top_k: int

    def __init__(
            self,
            *args,
            embedder: NodeEmbedder,
            threshold: float = None,
            top_k: int = None,
            **kwargs
    ):
        """

        """

        super().__init__(*args, **kwargs)
        self.embedder = embedder
        self.threshold = threshold
        self.top_k = top_k

        assert self.top_k is not None or self.threshold is not None, \
            'Must either have a top_k or threshold ofr the Contrastive Filter Validator.'

    def validate(
            self,
            graph: Graph,
            step: Step,
            new_premises: List[Node] = (),
            new_abductions: List[HyperNode] = (),
            new_deductions: List[HyperNode] = ()
    ) -> Tuple[List[Node], List[HyperNode], List[HyperNode]]:
        """

        """

        if len(new_deductions) == 0:
            return new_premises, new_abductions, new_deductions

        validated_new_deductions = []
        for deduction in new_deductions:
            encs = self.embedder.encode(deduction.nodes, normalize_values=True)
            rep = self.embedder.encode([graph[x] for x in deduction.arguments]).sum(0)

            curr_scores = [graph[x].scores.get('contrastive_d_score_total') for x in deduction.arguments if graph[x].scores.get('contrastive_d_score_total') is not None]
            curr_ns = [graph[x].scores.get('contrastive_d_score_n') for x in deduction.arguments if graph[x].scores.get('contrastive_d_score_n') is not None]

            raw_scores = torch.nn.functional.cosine_similarity(encs, rep, -1)
            for s in curr_scores:
                raw_scores += s
            scores = raw_scores / (len(curr_scores) + sum(curr_ns) + 1)


            valid_indices = list(range(len(deduction.nodes)))
            if self.threshold:
                valid_indices = torch.where(scores > self.threshold)[0].tolist()
                scores = scores[valid_indices]
            if self.top_k:
                valid_indices = scores.argsort(descending=True)
                scores = scores[valid_indices]
                valid_indices = valid_indices.tolist()
                valid_indices = valid_indices[0:min(len(valid_indices, self.top_k))] if len(valid_indices) > 0 else []

            valid_nodes = []
            for idx, v in enumerate(valid_indices):
                node = deduction.nodes[v]
                node.scores['contrastive_d_score'] = scores[idx].item()
                node.scores['contrastive_d_score_total'] = raw_scores[idx].item()
                node.scores['contrastive_d_score_n'] = len(curr_scores) + sum(curr_ns) + 1
                valid_nodes.append(node)
            deduction.nodes = valid_nodes
            validated_new_deductions.append(deduction)

        return new_premises, new_abductions, validated_new_deductions

    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        return self.search_obj_type, {
            'embedder': self.embedder.to_json_config(),
            'threshold': self.threshold,
            'top_k': self.top_k,
        }

    def to(self, device: str) -> 'AbductiveAgreementValidator':
        embedder = self.embedder.to(device)

        return AbductiveAgreementValidator(
            embedder=embedder,
            threshold=self.threshold,
            top_k=self.top_k
        )
