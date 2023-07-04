from multi_type_search.search.premise_retriever import PremiseRetriever
from multi_type_search.search.graph import Node
from multi_type_search.search.search_model import NodeEmbedder
from multi_type_search.search.search_model.types.contrastive.contrastive_utils import cosine_similarity_metric

from typing import List, Tuple, Dict, Union


class ContrastivePremiseRetriever(PremiseRetriever):
    search_obj_type = 'contrastive_premise_retriever'

    def __init__(
            self,
            node_embedder: NodeEmbedder,
            use_trajectories: bool = False,
            n: int = 25
    ):
        super().__init__(n)
        self.node_embedder = node_embedder
        self.use_trajectories = use_trajectories

    def reduce(self, premises: List[Node], target: Node, top_n: int = None, return_scores: bool = False) -> Union[List[Node], List[Tuple[Node, float]]]:
        if top_n is None:
            top_n = self.n

        if self.use_trajectories:
            return self.__reduce_trajectories__(premises, target, top_n, return_scores=return_scores)
        else:
            return self.__reduce_singular__(premises, target, top_n, return_scores=return_scores)

    def __reduce_trajectories__(self, premises: List[Node], target: Node, top_n: int, return_scores: bool = False) -> Union[List[Node], List[Tuple[Node, float]]]:
        embs = self.node_embedder.encode(premises)
        goal = self.node_embedder.encode([target])[0]

        trajectory_matrix = embs + embs.unsqueeze(1)

        trajectories = trajectory_matrix.reshape(-1, trajectory_matrix.shape[-1])
        goal_similarities = cosine_similarity_metric(trajectories, goal)

        top_scores = [(int(idx / embs.shape[0]), int(idx % embs.shape[0])) for idx in goal_similarities.argsort(descending=True).tolist()]
        top_premises = [x for y in top_scores for x in y]

        first_occurence = []
        if return_scores:
            [(first_occurence.append(x), top_scores[idx]) for x, idx in enumerate(top_premises) if x not in first_occurence]
            return [(premises[x[0]], x[1]) for x in first_occurence[0:top_n]]

        [first_occurence.append(x) for x, idx in enumerate(top_premises) if x not in first_occurence]
        return [premises[x] for x in first_occurence[0:top_n]]

    def __reduce_singular__(self, premises: List[Node], target: Node, top_n: int, return_scores: bool = False) -> Union[List[Node], List[Tuple[Node, float]]]:
        embs = self.node_embedder.encode(premises)
        goal = self.node_embedder.encode([target])[0]

        scores = cosine_similarity_metric(embs, goal)
        indices = scores.argsort(descending=True).tolist()

        if return_scores:
            return [(premises[x], scores[idx]) for idx, x in enumerate(indices[0:top_n])]
        return [premises[x] for idx, x in enumerate(indices[0:top_n])]

    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        return self.search_obj_type, {
            'node_embedder': self.node_embedder.to_json_config(),
            'n': self.n
        }
