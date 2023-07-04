from multi_type_search.search.comparison_metric import ComparisonMetric

from rouge_score.rouge_scorer import RougeScorer
from typing import List, Tuple, Dict


class RougeComparison(ComparisonMetric):
    """
    A wrapper comparison class around the Rouge score.  Returns the average F1 Score of each rouge type specified
    in the score() method.
    """

    search_obj_type: str = 'rouge_comparison'

    rouge_types: List[str]
    use_stemmer: bool
    scorer: RougeScorer

    def __init__(self, rouge_types: List[str], use_stemmer: bool = True):
        """
        :param rouge_types: A list of supported Rouge Types from the rouge_score package.
        :param use_stemmer: Whether or not to use the stemmer in the RougeScorer class.
        """

        assert (isinstance(rouge_types, list) or isinstance(rouge_types, tuple)) and len(rouge_types) > 0,\
            'Rouge scorer must have at least 1 type of scoring method.'

        self.rouge_types = rouge_types
        self.use_stemmer = use_stemmer
        self.scorer = RougeScorer(rouge_types, use_stemmer)

    def score(self, group: List[str], targets: List[str], **kwargs) -> List[float]:
        fmeasures = []

        # TODO - is there really no batching in rouge_scorer? there must be a way
        for target, item in zip(targets, group):
            scores = self.scorer.score(target, item)

            fmeasures.append(sum([scores[x].fmeasure for x in self.rouge_types]) / len(self.rouge_types))

        return fmeasures

    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        return self.search_obj_type, {
            'rouge_types': self.rouge_types,
            'use_stemmer': self.use_stemmer
        }
