from multi_type_search.search.step_selector import StepSelector, Step
from multi_type_search.search.graph import Graph

from typing import List, Tuple, Dict
import heapq
import json
import numpy as np
from pathlib import Path


class DPRSelector(StepSelector):
    """
    Step Selector using the DPR setup.  Requires the cached step embeddings file and the cached goal embeddings file.
    """
    search_obj_type: str = 'DPR_step_selector'

    iter_size: int
    step_list: List[Step]

    def __init__(self, steps_emb_file: str, goals_emb_file: str, iter_size: int = 1):
        """
        :param iter_size: How many steps to sample per iteration (DEFAULT 1)
        """

        self.iter_size = iter_size
        self.step_list = []

        self.steps_emb_file = Path(steps_emb_file)
        self.goals_emb_file = Path(goals_emb_file)

        self.steps_cache = json.load(self.steps_emb_file.open('r'))
        self.goals_cache = json.load(self.goals_emb_file.open('r'))

    def next(self) -> List[Step]:
        # Return the first self.iter_size steps or the entire list of self.step_list if its smaller.
        return [heapq.heappop(self.step_list) for _ in range(min([len(self.step_list), self.iter_size]))]

    def add_steps(self, steps: List[Step], graph: Graph = None) -> None:
        assert graph.goal.normalized_value in self.goals_cache, \
            'Graph goal needs to be in the cache.'

        # TODO - include the goal in the corpus?
        def create_step_text(x, reversed=False):
            return " ".join([graph[y].normalized_value for y in (x.arguments if not reversed else x.arguments[::-1])])

        goal_emb = np.array(self.goals_cache[graph.goal.normalized_value]).transpose()

        for idx, step in enumerate(steps):
            emb = self.steps_cache.get(create_step_text(step))
            if emb is None:
                emb = self.steps_cache.get(create_step_text(step, reversed=True))
                if emb is None:
                    print(f"WARNING STEP NOT FOUND: {step.arguments}")
                    continue

            teps[idx].score = -np.dot(goal_emb, np.array(emb))
            heapq.heappush(self.step_list, steps[idx])

    def __len__(self) -> int:
        return int(len(self.step_list) / self.iter_size) + (1 if len(self.step_list) % self.iter_size > 0 else 0)

    def reset(self) -> None:
        self.step_list = []

    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        return self.search_obj_type, {
            'iter_size': self.iter_size
        }
