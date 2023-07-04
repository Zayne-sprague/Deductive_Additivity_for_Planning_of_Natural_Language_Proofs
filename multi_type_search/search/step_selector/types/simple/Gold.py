from multi_type_search.search.step_selector import StepSelector, Step
from multi_type_search.search.graph import Graph
from multi_type_search.utils.paths import ROOT_FOLDER

from jsonlines import jsonlines
import json
from typing import List, Tuple, Dict


class GoldSelector(StepSelector):
    """
    Given the annotated graphs that have the gold deductions and arguments, this step selector will rank the closest
    gold step highest.  It will terminate the search when no more gold steps are available.
    """
    search_obj_type: str = 'gold_step_selector'

    iter_size: int
    step_list: List[Step]

    annotated_graphs: List[Graph]
    curr_graph: Graph
    curr_step: int

    def __init__(self, annotated_file: str, iter_size: int = 1):
        """
        :param iter_size: How many steps to sample per iteration (DEFAULT 1)
        """

        self.curr_step = 0
        self.curr_graph = None

        orig_data_file = ROOT_FOLDER / annotated_file
        if str(orig_data_file).endswith('.jsonl'):
            data = list(jsonlines.open(str(orig_data_file), 'r'))
        else:
            data = json.load(orig_data_file.open('r'))

        self.annotated_graphs = [Graph.from_json(x) for x in data]

        self.iter_size = iter_size
        self.step_list = []

    def next(self) -> List[Step]:
        self.curr_step += 1
        # Return the first self.iter_size steps or the entire list of self.step_list if its smaller.
        return [self.step_list.pop(0) for _ in range(min([len(self.step_list), self.iter_size]))]

    def add_steps(self, steps: List[Step], graph: Graph = None) -> None:
        if self.curr_graph is None:
            self.curr_graph = [x for x in self.annotated_graphs if x.goal.normalized_value == graph.goal.normalized_value and\
                                    len(x.premises) == len(graph.premises) and\
                                    len(set([y.normalized_value for y in x.premises]).difference([z.normalized_value for z in graph.premises])) == 0
                                ][0]

        if self.curr_step >= len(self.curr_graph.deductions):
            return

        if len(steps) == 0:
            return

        annotated_step = self.curr_graph.deductions[self.curr_step]

        steps[0].arguments = annotated_step.arguments
        self.step_list = [steps[0]]

    def __len__(self) -> int:
        return int(len(self.step_list) / self.iter_size) + (1 if len(self.step_list) % self.iter_size > 0 else 0)

    def reset(self) -> None:
        self.step_list = []
        self.curr_step = 0
        self.curr_graph = None


    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        return self.search_obj_type, {
            'iter_size': self.iter_size
        }
