from typing import Dict, Tuple, List
from abc import ABC, abstractmethod

from multi_type_search.search.search_object import SearchObject
from multi_type_search.search.graph import Node, HyperNode, Graph
from multi_type_search.search.step_selector import StepSelector, Step


class TerminationCriteria(SearchObject, ABC):

    @abstractmethod
    def should_terminate(
            self,
            new_premises: List[Node],
            new_abudctions: List[HyperNode],
            new_deductions: List[HyperNode],
            new_steps: List[Step],
            graph: Graph,
            step_selector: StepSelector
    ):
        raise NotImplemented("All termination criteria need to have 'should_terminate' defined.")

    def reset(self):
        return

    @classmethod
    def config_from_json(cls, type: str, arguments: Dict[str, any], device: str = 'cpu') -> 'SearchObject':
        from multi_type_search.search.termination_criteria import WallClockTimeTermination

        if type == WallClockTimeTermination.search_obj_type:
            return WallClockTimeTermination(**arguments)

        raise Exception(f'Unknown termination type: {type}')

    def to_json_config(self) -> Dict[str, any]:
        _type, args = self.__to_json_config__()
        return {
            'constructor_type': 'termination_criteria',
            'type': _type,
            'arguments': args
        }

    def to(self, device: str) -> 'SearchObject':
        return self
