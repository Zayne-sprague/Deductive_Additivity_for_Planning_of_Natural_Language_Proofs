from multi_type_search.search.comparison_metric import ComparisonMetric

from nltk.translate.bleu_score import sentence_bleu
from typing import List, Tuple, Dict, Callable


class BleuComparison(ComparisonMetric):
    """
    Bleu Metric for a group of sentences to a target.
    """

    search_obj_type: str = 'bleu_comparison'

    def score(
            self,
            group: List[str],
            targets: List[str],
            tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
            weights: List[float] = (0.25, 0.25, 0.25, 0.25)
    ) -> List[float]:
        """
        More info and description in comparison_metric.py

        :param group: The "left-hand side" of the comparison (what you are comparing)
        :param targets: The "right-hand side" of the comparison (what you are comparing to)
        :param tokenizer: Function that can take a string and return a list of string parts.
        :param weights: The N-Gram weights to use in the score (1.0) would be unigrams (0., 1.) would be bigrams only
            and (0.5, 0.5) would be unigrams + bigrams averaged. (So on so forth for higher order N-grams)
        :return: A list of scalars that represent some comparison between the group and target strings.
        """
        scores = []

        # TODO - is there really no batching in rouge_scorer? there must be a way
        for target, item in zip(targets, group):
            tokenized_item = tokenizer(item)
            tokenized_target = tokenizer(target)

            score = sentence_bleu([tokenized_item], tokenized_target, weights=weights)
            scores.append(score)

        return scores

    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        return self.search_obj_type, {}
