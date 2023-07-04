from multi_type_search.search.comparison_metric import ComparisonMetric

from nltk.translate.bleu_score import sentence_bleu
from typing import List, Tuple, Dict, Callable


class SelfBleuComparison(ComparisonMetric):
    """
    This is an implementation of Self Bleu.  Given a list of sentences (group) it will average the bleu score
    across all the other sentences in the group.  It will ignore the target list.

    you can read on what the Self-Bleu score is here -
    https://github.com/geek-ai/Texygen/blob/master/docs/evaluation.md#self-bleu-score
    """

    search_obj_type: str = 'self_bleu_comparison'


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
        for gidx, target in enumerate(group):

            others = [x for idx, x in enumerate(group) if gidx != idx]

            if len(others) == 0:
                return [0.,]

            try:
                score = sentence_bleu(
                    [tokenizer(x) for x in others],
                    tokenizer(target),
                    weights=weights
                )
            except Exception as e:
                print(f"Warning: SELF-BLEU errored: {e}")
                scores.append(0.)
                continue

            scores.append(score)

        return scores

    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        return self.search_obj_type, {}
