from multi_type_search.search.search_object import SearchObject
from multi_type_search.search.graph import Graph
from multi_type_search.search.step_selector import Step
from multi_type_search.search.step_type import StepType

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict


class StepValidator(SearchObject, ABC):
    """
    A Step Validator is used to determine if a given Step and it's arguments along with a Graph object are valid.  If it
    is determined that a step is invalid, the step is removed from the list.
    """

    @abstractmethod
    def validate(
            self,
            graph: Graph,
            unvalidated_steps: List[Step] = (),
    ) -> List[Step]:
        """
        Determines whether a list of steps are valid or not (according to the criteria of the specific StepValidator).

        :param graph: A graph with the nodes of the arguments in each step
        :param unvalidated_steps: Unvalidated steps
        :return: Validated steps
        """
        return potential_steps

    @classmethod
    def config_from_json(cls, type: str, arguments: Dict[str, any], device: str = 'cpu') -> 'StepValidator':
        from multi_type_search.search.step_validator import ConsanguinityThresholdStepValidator, \
            DepthThresholdStepValidator

        if arguments is None or len(arguments) == 0:
            arguments = {}

        if type == ConsanguinityThresholdStepValidator.search_obj_type:
            return ConsanguinityThresholdStepValidator(**arguments)
        if type == DepthThresholdStepValidator.search_obj_type:
            return DepthThresholdStepValidator(**arguments)

        raise Exception(f"Unknown Step Validator type given in config: {type}")

    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        raise NotImplemented("Please implement the __to_json_config__() for all step validators.")

    def to_json_config(self) -> Dict[str, any]:
        _type, args = self.__to_config__()
        return {
            'constructor_type': 'step_validator',
            'type': _type,
            'arguments': args
        }

    def to(self, device: str) -> 'StepValidator':
        return self
