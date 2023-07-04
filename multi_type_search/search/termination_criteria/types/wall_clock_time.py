from typing import Tuple, Dict, List
from timeit import default_timer as timer

from multi_type_search.search.termination_criteria import TerminationCriteria
from multi_type_search.search.graph import Node, HyperNode, Graph
from multi_type_search.search.step_selector import Step, StepSelector


class WallClockTimeTermination(TerminationCriteria):
    search_obj_type: str = 'wall_clock_time_termination'

    seconds_before_termination: float
    start: timer

    def __init__(
            self,
            seconds_before_termination: float = 60.,
    ):
        self.seconds_before_termination = float(seconds_before_termination)
        self.start = None

    def reset(self):
        self.start = timer()

    def should_terminate(
            self,
            new_premises: List[Node],
            new_abductions: List[HyperNode],
            new_deductions: List[HyperNode],
            new_steps: List[Step],
            graph: Graph,
            step_selector: StepSelector
    ):
        if self.start is None:
            return False

        if timer() - self.start >= self.seconds_before_termination:
            return True
        return False

    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        return self.search_obj_type, {
            'seconds_before_termination': self.seconds_before_termination
        }



