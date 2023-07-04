from multi_type_search.search.search_object import SearchObject
from multi_type_search.search.graph import Node

from typing import Dict, Tuple, List, Union
from abc import ABC, abstractmethod


class PremiseRetriever(SearchObject, ABC):
    """
    Selects a subset of premises given a list.  Can be incorporated into the main loop of the search algorithm

    (Typically useful for Task 3 like settings, large corpus of facts being reduced to a smaller set)
    """

    n: int

    def __init__(self, n: int = 25):
        """
        :param n: Number of premises to return when reduce() is called
        """

        self.n = n

    @abstractmethod
    def reduce(self, premises: List[Node], target: Node, top_n: int = None, return_scores: bool = False) -> Union[List[Node], List[Tuple[Node, float]]]:
        raise NotImplemented("All premise retrievers need the reduce() function.")

    @classmethod
    def config_from_json(cls, type: str, arguments: Dict[str, any], device: str = 'cpu') -> 'SearchModel':
        from multi_type_search.search.premise_retriever import BM25PremiseRetriever, ContrastivePremiseRetriever

        if type == BM25PremiseRetriever.search_obj_type:
            return BM25PremiseRetriever(**arguments)
        if type == ContrastivePremiseRetriever.search_obj_type:
            return ContrastivePremiseRetriever(**arguments)

        raise Exception(f'Unknown premise retriever type: {type}')

    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        raise NotImplemented("All premise retrievers need a __to_json_config__ definition.")

    def to_json_config(self) -> Dict[str, any]:
        _type, args = self.__to_json_config__()
        return {
            'constructor_type': 'premise_retriever',
            'type': _type,
            'arguments': args
        }

    def to(self, device: str) -> 'PremiseRetriever':
        return self
