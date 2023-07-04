from multi_type_search.search.comparison_metric import ComparisonMetric

from rouge_score.rouge_scorer import RougeScorer
from typing import List, Tuple, Dict


class ExactComparison(ComparisonMetric):
    """
    Simple comparison metric that looks for exact string matches (returning 1. when it does match and 0. otherwise)
    """

    search_obj_type: str = 'exact_comparison'

    def score(self, group: List[str], targets: List[str], **kwargs) -> List[float]:
        assert len(group) == len(targets), 'Pass the same length of strings in the group and target params'

        return [1. if x == y else 0. for x, y in zip(group, targets)]

    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        return self.search_obj_type, {}
