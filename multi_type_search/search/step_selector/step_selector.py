from multi_type_search.search.search_object import SearchObject
from multi_type_search.search.step_selector import Step
from multi_type_search.search.graph import Graph

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple


class StepSelector(SearchObject, ABC):
    """
    Class for selecting which step/s to take next in the search.  Also responsible for handling how to score/rank/store
    new steps created at each iteration of the search.

    This is meant to be a generic class that can be utilized as a template for various planner agents.
    """

    @abstractmethod
    def next(self) -> List[Step]:
        """
        This function returns a list (can be a list of 1) of steps to take in the next iteration of the search loop.

        :return: List of steps to take in the search loop.
        """
        raise NotImplementedError("Implement the next function for all step selectors")

    @abstractmethod
    def add_steps(self, steps: List[Step], graph: Graph) -> None:
        """
        This function is called at the end of a search iteration and is used to control how to add the new steps
            generated for that iteration.

        :param steps: A list of new steps generated from the previous search iteration
        :param graph: The graph that contains the arguments used in the list of Steps.
        """
        raise NotImplementedError("Implement how to handle new steps being added to the Step Selector.")

    @abstractmethod
    def __len__(self) -> int:
        """Returns the length of the Queue for the given step selector"""
        raise NotImplementedError("Make sure to implement a way to check for length.")

    def __iter__(self) -> 'StepSelector':
        return self

    def __next__(self) -> List[Step]:
        """Function so every step selector can function as an iterator"""
        try:
            result = self.next()
            assert isinstance(result, list)
            assert len(result) > 0
            assert result is not None
        except Exception:
            raise StopIteration

        return result

    @classmethod
    def config_from_json(cls, type: str, arguments: Dict[str, any], device: str = 'cpu') -> 'StepSelector':
        from multi_type_search.search.step_selector import BFSSelector, DFSSelector, GPT3Selector, \
            MultiLearnedCalibratorSelector, VectorSpaceSelector, BM25Selector, DPRSelector, GoldSelector

        if arguments is None or len(arguments) == 0:
            arguments = {}

        if type == BFSSelector.search_obj_type:
            return BFSSelector(**arguments)
        if type == DFSSelector.search_obj_type:
            return DFSSelector(**arguments)
        if type == GPT3Selector.search_obj_type:
            return GPT3Selector(**arguments)
        if type == MultiLearnedCalibratorSelector.search_obj_type:
            return MultiLearnedCalibratorSelector(**arguments)
        if type == VectorSpaceSelector.search_obj_type:
            return VectorSpaceSelector(**arguments)
        if type == BM25Selector.search_obj_type:
            return BM25Selector(**arguments)
        if type == DPRSelector.search_obj_type:
            return DPRSelector(**arguments)
        if type == GoldSelector.search_obj_type:
            return GoldSelector(**arguments)
        raise Exception(f"Unknown type of Step Selector in Config: {type} ")

    def reset(self) -> None:
        """Resets the step selector (empty queues etc.)"""
        pass

    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        raise NotImplemented("Please implement the __to_json_config__() func for all step selectors.")

    def to_json_config(self) -> Dict[str, any]:
        _type, args = self.__to_config__()
        return {
            'constructor_type': 'step_selector',
            'type': _type,
            'arguments': args
        }

    def to(self, device: str) -> 'StepSelector':
        return self
