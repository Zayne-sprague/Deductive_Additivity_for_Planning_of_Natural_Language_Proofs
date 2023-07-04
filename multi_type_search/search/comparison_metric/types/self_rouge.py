from multi_type_search.search.comparison_metric import ComparisonMetric, RougeComparison

from rouge_score.rouge_scorer import RougeScorer
from typing import List, Tuple, Dict


class SelfRougeComparison(ComparisonMetric):
    """
    This is an implementation of Self Bleu for Rouge.  Given a list of sentences (group) it will average the rouge score
    across all the other sentences in the group.  It will ignore the target list.

    you can read on what the Self-Bleu score is here -
    https://github.com/geek-ai/Texygen/blob/master/docs/evaluation.md#self-bleu-score
    """

    search_obj_type: str = 'self_rouge_comparison'

    rouge_comparison: RougeComparison

    def __init__(self, rouge_comparison: RougeComparison):
        """
        :param rouge_comparison: The RougeComparison metric for calculating the self-rouge scores
        """

        self.rouge_comparison = rouge_comparison

    def score(self, group: List[str], targets: List[str], **kwargs) -> List[float]:
        scores = []

        # TODO - is there really no batching in rouge_scorer? there must be a way
        for gidx, target in enumerate(group):

            others = [x for idx, x in enumerate(group) if gidx != idx]

            if len(others) == 0:
                return [0.] * len(group)

            individual_scores = self.rouge_comparison.score([target] * len(others), others)
            selfrouge_score = sum(individual_scores) / len(others)
            scores.append(selfrouge_score)

        return scores

    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        return self.search_obj_type, {
            'rouge_comparison': self.rouge_comparison.to_json_config(),
        }
