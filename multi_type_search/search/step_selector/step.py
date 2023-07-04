from dataclasses import dataclass, field
from typing import List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from multi_type_search.search.step_type.step_type import StepType


@dataclass
class Step:
    """
    This class acts as a way to represent how multiple generations/Nodes/HyperNodes etc. can be combined and with what
    type of model can combine them.  There is also a score attribute that allows for easy direct comparisons of steps
    based on some heuristic or model.
    """

    arguments: List[str] = field(compare=False)
    type: 'StepType' = field(compare=False)
    tmp: Dict[str, any] = field(compare=False, default_factory=dict)
    score: float = 0.

    def __eq__(self, other: 'Step') -> bool:
        """
        Comparison function for the Step Object

        Should only ever be compared with another step

        :param other: The other step to compare against
        :return: Boolean representing equality
        """

        same_args = " ".join(self.arguments) == " ".join(other.arguments)
        same_step_type = self.type.name == other.type.name
        same_score = self.score == other.score
        return same_args and same_step_type and same_score

    def __hash__(self):
        return hash((tuple(self.arguments), hash(self.type)))

    def __lt__(self, other) -> bool:
        return self.score < other.score

    def __gt__(self, other) -> bool:
        return self.score > other.score
