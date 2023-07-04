from multi_type_search.search.search_object import SearchObject
from multi_type_search.search.graph import Node, HyperNode, Graph
from multi_type_search.search.step_selector import Step

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict


class GenerationValidator(SearchObject, ABC):
    """
    This class will look at the new generations (Premise, Abductions, Deductions, etc.) and attempt to validate them
    and remove them if they are not.
    """

    def __init__(
            self,
            *args,
            **kwargs
    ):
        pass

    @abstractmethod
    def validate(
            self,
            graph: Graph,
            step: Step,
            new_premises: List[Node] = (),
            new_abductions: List[HyperNode] = (),
            new_deductions: List[HyperNode] = ()
    ) -> Tuple[List[Node], List[HyperNode], List[HyperNode]]:
        """
        This is the base class that will be called when new premises, abductions, or deductions are generated.

        :param graph: The graph with all the current search state
        :param step: The step that generated the outputs
        :param new_premises: New premises that were generated as an output of a step model.
        :param new_abductions: New abductions that were generated as an output of a step model.
        :param new_deductions: New deductions that were generated as an output of a step model.
        :return: A tuple of the filtered -- validated new premises, validated new abductions, validated new deductions
        """
        return new_premises, new_abductions, new_deductions

    @classmethod
    def config_from_json(cls, type: str, arguments: Dict[str, any], device: str = 'cpu') -> 'GenerationValidator':
        from multi_type_search.search.generation_validator import DeduplicationValidator, DeductiveAgreementValidator, \
            AbductiveAgreementValidator, HyperNodeOveralpValidator, ContrastiveFilterValidator

        if arguments is None or len(arguments) == 0:
            arguments = {}

        if type == AbductiveAgreementValidator.search_obj_type:
            return AbductiveAgreementValidator(**arguments)
        if type == DeductiveAgreementValidator.search_obj_type:
            return DeductiveAgreementValidator(**arguments)
        if type == DeduplicationValidator.search_obj_type:
            return DeduplicationValidator(**arguments)
        if type == HyperNodeOveralpValidator.search_obj_type:
            return HyperNodeOveralpValidator(**arguments)
        if type == ContrastiveFilterValidator.search_obj_type:
            return ContrastiveFilterValidator(**arguments)

        raise Exception(f"Unknown generation validators type in config: {type}")

    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        raise NotImplemented("All search models need a __to_json_config__ definition.")

    def to_json_config(self) -> Dict[str, any]:
        _type, args = self.__to_json_config__()
        return {
            'constructor_type': 'generation_validator',
            'type': _type,
            'arguments': args
        }

    def to(self, device: str) -> 'GenerationValidator':
        return self
