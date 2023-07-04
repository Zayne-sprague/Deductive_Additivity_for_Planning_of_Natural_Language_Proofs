from multi_type_search.search.comparison_metric import ComparisonMetric, RougeComparison, EntailmentComparison

from typing import List, Tuple, Dict


class RougeEntailmentHMComparison(ComparisonMetric):
    """
    Comparison metric that returns the harmonic mean of each score from a RougeComparison and EntailmentComparison
    metric for every group/target pair in the score() method.
    """

    search_obj_type: str = 'rouge_entailment_hm_comparison'

    rouge_comparison: RougeComparison
    entailment_comparison: EntailmentComparison

    def __init__(self, rouge_comparison: RougeComparison, entailment_comparison: EntailmentComparison):
        """
        :param rouge_comparison: A RougeComparison metric to use in the harmonic mean
        :param entailment_comparison: An EntailmentComparison metric to use in the harmonic mean
        """

        self.rouge_comparison = rouge_comparison
        self.entailment_comparison = entailment_comparison

    def score(self, group: List[str], targets: List[str], **kwargs) -> List[float]:
        e_score = self.entailment_comparison.score(group, targets)
        r_score = self.rouge_comparison.score(group, targets)

        return [(2 * e * r) / (e + r) for e, r in zip(e_score, r_score)]

    def to(self, device: str) -> 'RougeEntailmentHMComparison':
        entailment_comparison = self.entailment_comparison.to(device)
        return RougeEntailmentHMComparison(
            self.rouge_comparison,
            entailment_comparison
        )

    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        return self.search_obj_type, {
            'rouge_comparison': self.rouge_comparison.to_json_config(),
            'entailment_comparison': self.entailment_comparison.to_json_config()
        }
