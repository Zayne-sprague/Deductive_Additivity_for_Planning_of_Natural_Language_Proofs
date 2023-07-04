from multi_type_search.search.search_object import SearchObject

from typing import Dict, Tuple


class SearchModel(SearchObject):
    """
    Search models are pytorch implementations of networks that are used in different ways for the search.  Since they
    share similar architectures/code they are put together under the SearchModels class (although they may do vastly
    different things at different parts of the search.)
    """

    @classmethod
    def config_from_json(cls, type: str, arguments: Dict[str, any], device: str = 'cpu') -> 'SearchModel':
        from multi_type_search.search.search_model import EntailmentModel, CalibratorHeuristic, StepModel, NodeEmbedder

        if type == EntailmentModel.search_obj_type:
            return EntailmentModel(**arguments, device=device)
        if type == CalibratorHeuristic.search_obj_type:
            return CalibratorHeuristic(**arguments, device=device)
        if type == StepModel.search_obj_type:
            return StepModel(**arguments, device=device)
        if type == NodeEmbedder.search_obj_type:
            return NodeEmbedder(**arguments, device=device)

        raise Exception(f'Unknown search model type: {type}')

    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        raise NotImplemented("All search models need a __to_json_config__ definition.")

    def to_json_config(self) -> Dict[str, any]:
        _type, args = self.__to_json_config__()
        return {
            'constructor_type': 'search_model',
            'type': _type,
            'arguments': args
        }

    def to(self, device: str) -> 'SearchModel':
        return self
