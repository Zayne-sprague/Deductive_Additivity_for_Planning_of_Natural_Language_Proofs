from typing import Dict, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod


class ConstructorTypes:
    """List of allowed constructor types of search objects."""
    step_selector = 'step_selector'
    step_type = 'step_type'
    step_validator = 'step_validator'
    generation_validator = 'generation_validator'
    search_model = 'search_model'
    comparison_metric = 'comparison_metric'
    termination_criteria = 'termination_criteria'
    premise_retriever = 'premise_retriever'


class SearchObject(ABC):
    """
    Fundamental object class for things related to the search.  This allows for some simple standardization mostly for
    reading in or writing to config formats.
    """

    @classmethod
    def from_config(
            cls,
            constructor_type: str,
            constructor: Dict[str, any],
            device: str = 'cpu'
    ) -> 'SearchObject':
        """Helper function for creating a Search Object from a config"""

        return cls.__from_config__(constructor_type, constructor.get('type'), constructor.get('arguments'), device=device)

    @classmethod
    def __from_config__(
            cls,
            constructor_type: str,
            type: str,
            arguments: Dict[str, any],
            device: str = 'cpu'
    ) -> 'SearchObject':
        """
        Recursive call that instantiates search objects in an argument block for a top level search object.  I.E. if
        an argument is itself another search object, that search object must be instantiated and assigned before we can
        create the higher level search object.

        :param constructor_type: T
        :param type:
        :param arguments:
        :param device:
        :return:
        """
        from multi_type_search.search.step_type import StepType
        from multi_type_search.search.step_selector import StepSelector
        from multi_type_search.search.step_validator import StepValidator
        from multi_type_search.search.search_model import SearchModel
        from multi_type_search.search.generation_validator import GenerationValidator
        from multi_type_search.search.comparison_metric import ComparisonMetric
        from multi_type_search.search.termination_criteria import TerminationCriteria
        from multi_type_search.search.premise_retriever import PremiseRetriever

        constructed_args = {}

        if arguments is not None:
            for arg, item in arguments.items():
                if isinstance(item, dict) and item.get('type') is not None:
                    # TODO - this shouldn't be here. (config overwrites code? or should code overwrite config?)
                    arg_device = device
                    if item.get('arguments') is not None and 'device' in item.get('arguments', {}):
                        arg_device = item['arguments']['device']
                        del item['arguments']['device']

                    constructed_args[arg] = cls.__from_config__(
                        item.get('constructor_type'),
                        item.get('type'),
                        item.get('arguments'),
                        arg_device
                    )
                else:
                    constructed_args[arg] = item

        if constructor_type == ConstructorTypes.step_selector:
            return StepSelector.config_from_json(type, constructed_args, device=device)
        if constructor_type == ConstructorTypes.step_type:
            return StepType.config_from_json(type, constructed_args, device=device)
        if constructor_type == ConstructorTypes.step_validator:
            return StepValidator.config_from_json(type, constructed_args, device=device)
        if constructor_type == ConstructorTypes.search_model:
            return SearchModel.config_from_json(type, constructed_args, device=device)
        if constructor_type == ConstructorTypes.generation_validator:
            return GenerationValidator.config_from_json(type, constructed_args, device=device)
        if constructor_type == ConstructorTypes.comparison_metric:
            return ComparisonMetric.config_from_json(type, constructed_args, device=device)
        if constructor_type == ConstructorTypes.termination_criteria:
            return TerminationCriteria.config_from_json(type, constructed_args, device=device)
        if constructor_type == ConstructorTypes.premise_retriever:
            return PremiseRetriever.config_from_json(type, constructed_args, device=device)

    @classmethod
    @abstractmethod
    def config_from_json(cls, type: str, arguments: Dict[str, any], device: str = 'cpu') -> 'SearchObject':
        """
        All search objects need to be able to be instantiated from json to allow for easy experiments that are
        completely controlled via a config file.  A typical configuration in json would look like

        {
            "constructor_type": {search_object_base_class}
            "type": {search_object_type, i.e. "DFS" for step_selector},
            "arguments": {
                "arg1": 1,
                "arg2": 2,
                "arg3": {
                    "constructor_type": ...
                    ...
                }
            }
        }

        This configuration allows for nested search objects to substantiated and passed as arguments for complex
        objects.

        :param type: Search object type, for example "DFS" for the step_selector search type constructor
        :param arguments: A list of arguments needed to create the search object.
        :param device: The device we would like to put the search object on (optional, default will be CPU)
        :return: The created search object.
        """

        raise NotImplemented("All search objects need to be able to be instantiated via json")

    @abstractmethod
    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        """Helper function for converting a search object into its corresponding config json."""
        raise NotImplemented("All search objects should be able to be written out as a json config")

    @abstractmethod
    def to_json_config(self) -> Dict[str, any]:
        """Creates a JSON configuration for the current search object that can be reloaded when needed."""
        raise NotImplemented("All search objects should be able to be written out as a json config")

    @abstractmethod
    def to(self, device: str) -> 'SearchObject':
        """Allows a Search Object to be transferred to a new torch device."""
        return self
