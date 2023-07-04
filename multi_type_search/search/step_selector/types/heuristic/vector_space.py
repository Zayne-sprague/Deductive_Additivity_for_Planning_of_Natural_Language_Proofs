import heapq

from multi_type_search.search.step_selector import StepSelector, Step
from multi_type_search.search.step_type import StepTypes
from multi_type_search.search.graph import Node, Graph, GraphKeyTypes, decompose_index
from multi_type_search.utils.paths import SEARCH_FOLDER
from multi_type_search.search.search_model import NodeEmbedder, contrastive_utils


from typing import List, Tuple, Dict, Optional
import torch


def create_trajectories(x, y, method: str = 'add'):
    # TODO - dupe code.
    if method == 'add':
        return x + y
    if method == 'subtract':
        return x - y
    if method == 'multiply':
        return x * y
    if method == 'max_pool':
        return torch.max(torch.stack([x, y], dim=0), dim=0)[0]
    if method == 'min_pool':
        return torch.min(torch.stack([x, y], dim=0), dim=0)[0]
    if method == 'avg_pool':
        return torch.mean(torch.stack([x, y], dim=0), dim=0)


class VectorSpaceSelector(StepSelector):
    """
    """
    search_obj_type: str = 'vector_space_step_selector'

    iter_size: int
    abductive_queue: List[Step]
    deductive_queue: List[Step]
    last_step_type: Optional[StepTypes]

    abductive_points: torch.Tensor
    deductive_points: torch.Tensor

    model: NodeEmbedder

    def __init__(
            self,
            model: NodeEmbedder,
            deductive_nms_threshold: float = 0.005,
            abductive_nms_threshold: float = 0.1,
            iter_size: int = 1,
            log_diffs: bool = False,
            use_norm: bool = True,
            redo_all_steps: bool = False,
            similarity_metric: str = 'cosine',
            use_nms: bool = False,
            weight_by_rep_score: bool = False,
            trajectory_creation_method: str = 'add'
    ):
        """
        :param iter_size: How many steps to sample per iteration (DEFAULT 1)
        """

        self.log_diffs = log_diffs

        self.iter_size = iter_size
        self.model = model

        self.abductive_queue = []
        self.deductive_queue = []
        self.last_step_type = None

        self.abductive_points = None
        self.deductive_points = None

        self.use_norm = use_norm
        self.similarity_metric = similarity_metric
        self.redo_all_steps = redo_all_steps

        self.deductive_nms_threshold = deductive_nms_threshold
        self.abductive_nms_threshold = abductive_nms_threshold

        self.use_nms = use_nms
        self.weight_by_rep_score = weight_by_rep_score

        self.trajectory_creation_method = trajectory_creation_method

        # --- #
        self.g = None
        self.ds = None

    def next(self) -> List[Step]:
        # Return the first self.iter_size steps or the entire list of self.step_list if its smaller.
        if (self.last_step_type != StepTypes.Deductive or self.abductive_queue_len == 0) and self.deductive_queue_len > 0:
            self.last_step_type = StepTypes.Deductive
            steps = [heapq.heappop(self.deductive_queue) for _ in range(min([len(self.deductive_queue), self.iter_size]))]

            if self.use_nms:
                for step in steps:
                    if self.deductive_points is None:
                        self.deductive_points = step.tmp['point']
                    else:
                        self.deductive_points = torch.cat([self.deductive_points, step.tmp['point']])
            return steps
        elif (self.last_step_type != StepTypes.Abductive or self.deductive_queue_len == 0) and self.abductive_queue_len > 0:
            self.last_step_type = StepTypes.Abductive
            steps = [heapq.heappop(self.abductive_queue) for _ in range(min([len(self.abductive_queue), self.iter_size]))]

            if self.use_nms:
                for step in steps:
                    if self.abductive_points is None:
                        self.abductive_points = step.tmp['point'].unsqueeze(0)
                    else:
                        self.abductive_points = torch.cat([self.abductive_points, step.tmp['point'].unsqueeze(0)])
            return steps
        else:
            return []

    def compute_similarity(self, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        if self.similarity_metric == 'euclidean_distance':
            sim = contrastive_utils.l2_simularity_metric(e1, e2)
        elif self.similarity_metric == 'cosine':
            sim = contrastive_utils.cosine_similarity_metric(e1, e2)
            # Cosine similarity is also from [-1, 1] where -1 indicates vastly different vectors -- so negate it :)
            sim = -sim
        else:
            raise Exception(f"Unknown similarity metric : {self.similarity_metric}")

        return sim

    def add_steps(self, steps: List[Step], graph: Graph = None) -> None:
        if self.redo_all_steps:
            steps = [*steps, *self.abductive_queue, *self.deductive_queue]
            self.abductive_queue = []
            self.deductive_queue = []

        if self.g is None or self.ds is None:
            self.g = self.model.encode([graph.goal], normalize_values=True).squeeze()
            self.ds = self.model.encode(graph.premises, normalize_values=True)

        nodes = [graph[x] for step in steps for x in step.arguments]
        if len(nodes) == 0:
            return

        encs = self.model.encode(nodes, normalize_values=True)
        pair_encs = encs.view([-1, 2, encs.shape[-1]])

        if steps[0].type.name == StepTypes.Deductive:
            trajectories = create_trajectories(pair_encs[:, 0], pair_encs[:, 1], self.trajectory_creation_method)

            sims = None
            if self.use_nms and self.deductive_points is not None:
                BATCH_SIZE = 150
                for i in range(0, trajectories.shape[0], BATCH_SIZE):
                    end_idx = min([trajectories.shape[0], i + BATCH_SIZE])

                    if sims is not None:
                        sims = torch.cat([sims, self.compute_similarity(self.deductive_points, trajectories.unsqueeze(1)[i:end_idx]).min(axis=1).values])
                    else:
                        sims = self.compute_similarity(self.deductive_points, trajectories.unsqueeze(1)[i:end_idx]).min(axis=1).values

                # sims = self.compute_similarity(self.deductive_points, trajectories.unsqueeze(1)).min(axis=1)
                invalid_vals = sims <= self.deductive_nms_threshold

                trajectories = trajectories[invalid_vals == False]
                if trajectories.shape[0] == 0:
                    return

                valid_indices = torch.arange(0, invalid_vals.shape[-1])[invalid_vals == False]
                steps = [steps[idx] for idx in valid_indices.tolist()]

            goal_sims = self.compute_similarity(trajectories, self.g)

            for idx, step in enumerate(steps):
                if self.use_nms:
                    step.tmp['point'] = torch.stack([trajectories[idx]])
                step.score = goal_sims[idx].detach().item()

                if self.weight_by_rep_score:
                    # denom = 0
                    # for arg in step.arguments:
                    #     mk,didx,_ = decompose_index(arg)
                    #     if mk == GraphKeyTypes.DEDUCTIVE:
                    #         dargs = graph.deductions[didx].arguments
                    #         denc = self.model.encode([graph[arg]]).squeeze()
                    #         rep = self.model.encode([graph[x] for x in dargs]).sum(0)
                    #         score = self.compute_similarity(denc, rep)
                    #         step.score += score.detach().item()
                    #         denom += 1
                    # if denom > 0:
                    #     step.score /= denom
                    scores = [graph[x].scores.get('contrastive_d_score') for x in step.arguments if graph[x].scores.get('contrastive_d_score') is not None]
                    if len(scores) > 0:
                        step.score = sum([step.score, *scores]) / (len(scores) + 2)
                heapq.heappush(self.deductive_queue, step)

        elif steps[0].type.name == StepTypes.Abductive:
            for step in steps:
                loc = points[1, :] - points[0, :]
                if self.use_norm:
                    loc = contrastive_utils.spherical_norm(loc)

                if self.abductive_points is not None and self.use_nms:
                    loc_difference = self.compute_similarity(self.abductive_points, loc).min()
                    if loc_difference <= self.abductive_nms_threshold:
                        continue

                mn_score = None
                for idx in range(ds.shape[0]):
                    s = self.compute_similarity(loc, ds[idx, :])
                    if mn_score is None or s < mn_score:
                        mn_score = s

                if mn_score is not None:
                    step.tmp['point'] = loc

                    step.score = mn_score.detach().item()
                    heapq.heappush(self.abductive_queue, step)

        else:
            raise Exception("Unknown step type")


    @property
    def abductive_queue_len(self) -> int:
        """
        Helper attribute for returning the length of the abductive queue
        :return: Length of the Abductive Queue
        """

        return int(len(self.abductive_queue) / self.iter_size) + (1 if len(self.abductive_queue) % self.iter_size > 0 else 0)

    @property
    def deductive_queue_len(self) -> int:
        """
        Helper attribute for returning the length of the deductive queue
        :return: Length of the Deductive Queue
        """

        return int(len(self.deductive_queue) / self.iter_size) + (1 if len(self.deductive_queue) % self.iter_size > 0 else 0)

    def __len__(self) -> int:
        return self.abductive_queue_len + self.deductive_queue_len

    def reset(self) -> None:
        self.abductive_queue = []
        self.deductive_queue = []
        self.deductive_points = None
        self.abductive_points = None

        self.g = None
        self.ds = None


    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        return self.search_obj_type, {
            'iter_size': self.iter_size,
            'model': self.model.to_json_config(),
            'deductive_nms_threshold': self.deductive_nms_threshold,
            'abductive_nms_threshold': self.abductive_nms_threshold,
            'harmonic_mean': self.harmonic_mean,
            'log_diffs': self.log_diffs,
            'use_norm': self.use_norm,
            'similarity_metric': self.similarity_metric,
            'redo_all_steps': self.redo_all_steps,
            'use_nms': self.use_nms,
        }
