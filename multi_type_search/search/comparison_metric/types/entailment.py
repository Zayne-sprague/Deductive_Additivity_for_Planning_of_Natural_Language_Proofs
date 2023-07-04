from multi_type_search.search.comparison_metric import ComparisonMetric
from multi_type_search.search.search_model import EntailmentModel

from typing import Union, List, Dict, Tuple
from enum import Enum


class EntailmentMethod(Enum):
    """
    Selection of different entailment comparison metrics.

    group_to_target -> returns the entailment scores of the group/target pair as ENTAILS(group -> target)
    target_to_group -> returns the entailment scores of the group/target pair as ENTAILS(target -> group)
    mutual -> returns the entailment scores of the group/target pair as (group_to_target + target_to_group) / 2
    max -> returns the entailment scores of the group/target pair as MAX(group_to_target, target_to_group)
    """

    group_to_target = 'group_to_target'
    target_to_group = 'target_to_group'
    mutual = 'mutual'
    max = 'max'


class EntailmentComparison(ComparisonMetric):
    """
    A comparison metric that uses an entailment model to compare two strings.

    Only the Entailment Score is returned in the score() method for every string group/target pair.
    """

    search_obj_type: str = 'entailment_comparison'

    entailment_model: EntailmentModel
    entailment_method: EntailmentMethod

    def __init__(self, entailment_model: EntailmentModel, entailment_method: EntailmentMethod):
        """
        :param entailment_model: The PyTorch SearchModel implementation of the Entailment Model
        :param entailment_method: The Entailment method when scoring a group/target pair.
        """

        self.entailment_model = entailment_model
        self.entailment_method = entailment_method if isinstance(entailment_method, EntailmentMethod) else EntailmentMethod[entailment_method]

    def score(self, group: List[str], targets: List[str], **kwargs) -> List[float]:
        assert len(group) == len(targets), 'the number of items in the group must match the number of targets.'

        if self.entailment_method == EntailmentMethod.target_to_group:
            return self.entailment_model.score(predictions=targets, targets=group)
        elif self.entailment_method == EntailmentMethod.group_to_target:
            return self.entailment_model.score(predictions=group, targets=targets)
        elif self.entailment_method == EntailmentMethod.mutual:
            ttg = self.entailment_model.score(predictions=targets, targets=group)
            gtt = self.entailment_model.score(predictions=group, targets=targets)
            return [(x + y) / 2 for x, y in zip(ttg, gtt)]
        elif self.entailment_method == EntailmentMethod.max:
            ttg = self.entailment_model.score(predictions=targets, targets=group)
            gtt = self.entailment_model.score(predictions=group, targets=targets)
            return [max(x, y) for x, y in zip(ttg, gtt)]
        else:
            raise Exception(f"Unknown entailment type: {self.entailment_method}")

    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        return self.search_obj_type, {
            'entailment_model': self.entailment_model.to_json_config(),
            'entailment_method': self.entailment_method.value,
        }

    def to(self, device: str) -> 'EntailmentComparison':
        entailment_model = self.entailment_model.to(device)
        return EntailmentComparison(
            entailment_model,
            self.entailment_method
        )
