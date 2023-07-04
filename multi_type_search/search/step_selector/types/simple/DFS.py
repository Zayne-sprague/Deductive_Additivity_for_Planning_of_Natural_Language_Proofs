from multi_type_search.search.step_selector import StepSelector, Step
from multi_type_search.search.graph import Graph

from typing import List, Tuple, Dict


class DFSSelector(StepSelector):
    """
    Depth First Search Selector
    A simple step selector that will always favor deeper steps (newer steps).
    """

    search_obj_type: str = 'DFS_step_selector'

    iter_size: int
    step_list: List[Step]

    def __init__(self, iter_size: int = 1):
        """
        :param iter_size: How many steps to sample per iteration (DEFAULT 1)
        """

        self.iter_size = iter_size
        self.step_list = []

    def next(self) -> List[Step]:
        # Return the first self.iter_size steps or the entire list of self.step_list if its smaller.
        return [self.step_list.pop(0) for _ in range(min([len(self.step_list), self.iter_size]))]

    def add_steps(self, steps: List[Step], graph: Graph = None) -> None:
        self.step_list = [*steps, *self.step_list]

    def __len__(self) -> int:
        return int(len(self.step_list) / self.iter_size) + (1 if len(self.step_list) % self.iter_size > 0 else 0)

    def reset(self) -> None:
        self.step_list = []

    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        return self.search_obj_type, {
            'iter_size': self.iter_size
        }
