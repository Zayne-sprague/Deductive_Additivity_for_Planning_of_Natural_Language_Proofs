from multi_type_search.search.premise_retriever import PremiseRetriever
from multi_type_search.search.graph import Node

import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Tuple, Dict, Union


class BM25PremiseRetriever(PremiseRetriever):
    search_obj_type = 'bm25_premise_retriever'

    def __init__(self, n: int = 25):
        super().__init__(n)
        self.bm25_model = BM25Okapi

    def __tokenizer__(self, x: Node):
        """TODO - should this be a better tokenizer?"""
        val = x.normalized_value
        if val.endswith('.'):
            val = val[:-1]
        val = val.split(" ")
        return val

    def reduce(self, premises: List[Node], target: Node, top_n: int = None, return_scores: bool = False) -> Union[List[Node], List[Tuple[Node, float]]]:
        if top_n is None:
            top_n = self.n

        model = self.bm25_model([self.__tokenizer__(x) for x in premises])
        result = model.get_top_n(self.__tokenizer__(target), [x.normalized_value for x in premises], top_n)
        norms = [x.normalized_value for x in premises]
        indices = [norms.index(x) for x in result]

        if return_scores:
            scores = model.get_scores(self.__tokenizer__(target))
            scores = np.sort(scores)[::-1]
            return [(premises[x], scores[idx]) for idx, x in enumerate(indices)]
        return [premises[x] for x in indices]

    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        return self.search_obj_type, {
            'n': self.n
        }
