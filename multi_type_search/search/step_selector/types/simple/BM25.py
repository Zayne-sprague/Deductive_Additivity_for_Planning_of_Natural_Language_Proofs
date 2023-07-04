from multi_type_search.search.step_selector import StepSelector, Step
from multi_type_search.search.graph import Graph

from typing import List, Tuple, Dict
from rank_bm25 import BM25Okapi, BM25L, BM25Plus
import heapq


class BM25Selector(StepSelector):
    """
    Step Selector that uses BM25 to find lexically overlapping pairs of premises with the graphs goal.
    """
    search_obj_type: str = 'BM25_step_selector'

    iter_size: int
    step_list: List[Step]

    def __init__(self, iter_size: int = 1):
        """
        :param iter_size: How many steps to sample per iteration (DEFAULT 1)
        """

        self.iter_size = iter_size
        self.step_list = []
        self.base_model = BM25Okapi
        # self.base_model = BM25L
        # self.base_model = BM25Plus

    def next(self) -> List[Step]:
        # Return the first self.iter_size steps or the entire list of self.step_list if its smaller.
        return [heapq.heappop(self.step_list) for _ in range(min([len(self.step_list), self.iter_size]))]

    def add_steps(self, steps: List[Step], graph: Graph = None) -> None:
        # TODO - include the goal in the corpus?
        step_text = [" ".join([graph[y].normalized_value for y in x.arguments]).split() for x in steps]
    
        if len(step_text) == 0:
            return

        model = self.base_model(step_text)

        scores = model.get_scores(graph.goal.normalized_value.replace("\t", " ").split(" "))

        for idx, step in enumerate(steps):
            step.score = -scores[idx]

            heapq.heappush(self.step_list, step)

    def __len__(self) -> int:
        return int(len(self.step_list) / self.iter_size) + (1 if len(self.step_list) % self.iter_size > 0 else 0)

    def reset(self) -> None:
        self.step_list = []

    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        return self.search_obj_type, {
            'iter_size': self.iter_size
        }
