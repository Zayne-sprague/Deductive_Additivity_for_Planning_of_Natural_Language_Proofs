from multi_type_search.search.graph import Graph
from multi_type_search.search.step_selector import StepSelector, Step
from multi_type_search.search.step_type import StepTypes
from multi_type_search.search.search_model import CalibratorHeuristic

import heapq
from typing import List, Tuple, Dict, Optional


class MultiLearnedCalibratorSelector(StepSelector):
    """
    This StepSelector uses a CalibratedHeuristic SearchModel object, score each deductive/abductive step and store them
    in separate queues.  Each queue is then alternated between for each next step, if one is empty the other will be
    used solely until both are exhausted.
    """

    search_obj_type: str = 'calibrator_heuristic_step_selector'

    abductive_queue: List[Step]
    deductive_queue: List[Step]
    last_step_type: Optional[StepTypes]

    def __init__(
            self,
            abductive_heuristic_model: CalibratorHeuristic,
            deductive_heuristic_model: CalibratorHeuristic,
            iter_size: int = 1,
    ):
        """
        :param abductive_heuristic_model: The abductive Calibrator Heuristic Search Model object
        :param deductive_heuristic_model: The deductive Calibrator Heuristic Search Model object.
        :param iter_size: How many steps to sample per iteration (DEFAULT 1)
        """

        self.iter_size = iter_size
        self.step_list = []
        self.last_step_type = None

        self.abductive_heuristic_model = abductive_heuristic_model
        self.deductive_heuristic_model = deductive_heuristic_model

        self.abductive_queue = []
        self.deductive_queue = []

    def next(self) -> List[Step]:
        # Alternates between abductive and deductive steps in the queue since there are two models using to score each
        # respectively (comparing their scores directly means nothing).
        if (self.last_step_type != StepTypes.Deductive or self.abductive_queue_len == 0) and self.deductive_queue_len > 0:
            self.last_step_type = StepTypes.Deductive
            return [heapq.heappop(self.deductive_queue) for _ in range(min([len(self.deductive_queue), self.iter_size]))]
        elif (self.last_step_type != StepTypes.Abductive or self.deductive_queue_len == 0) and self.abductive_queue_len > 0:
            self.last_step_type = StepTypes.Abductive
            return [heapq.heappop(self.abductive_queue) for _ in range(min([len(self.abductive_queue), self.iter_size]))]
        else:
            return []

    def add_steps(self, steps: List[Step], graph: Graph) -> None:
        abductive_steps = []
        deductive_steps = []

        for idx, step in enumerate(steps):
            step_type = step.type.name

            if step_type == StepTypes.Abductive:
                abductive_steps.append(step)
            elif step_type == StepTypes.Deductive:
                deductive_steps.append(step)
            else:
                raise Exception("Unknown step type given to the multi_type_learned step selector!")

        a_scores = self.abductive_heuristic_model.score_steps(graph, abductive_steps)
        d_scores = self.deductive_heuristic_model.score_steps(graph, deductive_steps)

        for step, score in zip(abductive_steps, a_scores):
            step.score = -score
            heapq.heappush(self.abductive_queue, step)

        for step, score in zip(deductive_steps, d_scores):
            step.score = -score
            heapq.heappush(self.deductive_queue, step)

    @property
    def abductive_queue_len(self) -> int:
        """
        Helper attribute for returning the length of the abductive queue
        :return: Length of the Abductive Queue
        """

        return int(len(self.abductive_queue) / self.iter_size) + (1 if len(self.abductive_queue) % self.iter_size > 0 else 0)

    @property
    def deductive_queue_len(self) -> int:
        """
        Helper attribute for returning the length of the deductive queue
        :return: Length of the Deductive Queue
        """

        return int(len(self.deductive_queue) / self.iter_size) + (1 if len(self.deductive_queue) % self.iter_size > 0 else 0)

    def __len__(self) -> int:
        return self.abductive_queue_len + self.deductive_queue_len

    def reset(self) -> None:
        self.abductive_queue = []
        self.deductive_queue = []

    def to(self, device: str) -> 'StepSelector':
        abductive_calibrator = self.abductive_heuristic_model.to(device)
        deductive_calibrator = self.deductive_heuristic_model.to(device)

        return MultiLearnedCalibratorSelector(
            abductive_calibrator,
            deductive_calibrator,
            self.iter_size
        )

    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        return self.search_obj_type, {
            'abductive_heuristic_model': self.abductive_heuristic_model.to_json_config(),
            'deductive_heuristic_model': self.deductive_heuristic_model.to_json_config(),
            'iter_size': self.iter_size,

        }

