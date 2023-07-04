from typing import List
from tqdm import tqdm
import pickle
from pathlib import Path
import torch
from copy import deepcopy

from multi_type_search.search.graph import Graph
from multi_type_search.search.search_model import NodeEmbedder
from multi_type_search.search.search_model.types.contrastive import contrastive_utils
from multi_type_search.utils.paths import ENTAILMENT_BANK_FOLDER
from multi_type_search.scripts.create_shallow_graphs import create_shallow_graphs


def build_embedding_table(
        graphs: List[Graph],
        node_embedder: NodeEmbedder,
        include_distractors: bool = False,
        disable_tqdm: bool = False
):
    facts = {}
    goals = {}

    for graph in tqdm(graphs, desc='building embedding table', total=len(graphs), disable=disable_tqdm):
        to_encode = [*graph.premises]
        if include_distractors:
            to_encode.extend(graph.distractor_premises)

        for premise in to_encode:
            if premise.normalized_value in facts:
                continue
            facts[premise.normalized_value] = node_embedder.encode([premise]).squeeze()

        if graph.goal.normalized_value not in goals:
            goals[graph.goal.normalized_value] = node_embedder.encode([graph.goal]).squeeze()

    return facts, goals


def t3_step_recall_with_map_one_shot(
        fact_map,
        goal_map,
        graphs: List[Graph],
        disable_tqdm: bool = False
):
    facts = torch.stack(list(fact_map.values()))
    goals = torch.stack(list(goal_map.values()))

    trajectories = facts + facts.unsqueeze(1)
    trajectories = trajectories.reshape(-1, trajectories.shape[-1])

    # Cosine Similarity
    goal_similarities = ((trajectories.unsqueeze(1) * goals).sum(2) / (torch.norm(trajectories) * torch.norm(goals))).T
    sorted_indices = goal_similarities.argsort(descending=True).tolist()

    fact_map_len = len(fact_map)
    sorted_fxfs = [[(idx // fact_map_len, idx % fact_map_len) for idx in x] for x in sorted_indices]

    fact_strings = list(fact_map.keys())
    goal_strings = list(goal_map.keys())

    ranks = []

    for graph in tqdm(graphs, desc='Running T3 step recall exp', total=len(graphs), disable=disable_tqdm):
        correct_indices = tuple(fact_strings.index(x.normalized_value) for x in graph.premises)
        idx = goal_strings.index(graph.goal.normalized_value)
        rank = sorted_fxfs[idx].index(correct_indices)
        ranks.append(rank)

    mrr = sum([1 / (x+1) for x in ranks]) / max(1, len(ranks))
    return mrr, ranks


def t3_step_recall_with_map(
        fact_map,
        goal_map,
        graphs: List[Graph],
        disable_tqdm: bool = False
):
    facts = torch.stack(list(fact_map.values()))
    goals = torch.stack(list(goal_map.values()))

    trajectories = facts + facts.unsqueeze(1)
    trajectories = trajectories.reshape(-1, trajectories.shape[-1])
    normed_trajectories = torch.norm(trajectories)

    fact_map_len = len(fact_map)
    fact_strings = list(fact_map.keys())
    goal_strings = list(goal_map.keys())

    ranks = []

    for graph in tqdm(graphs, desc='Running T3 step recall exp', total=len(graphs), disable=disable_tqdm):
        idx = goal_strings.index(graph.goal.normalized_value)

        goal_similarities = (trajectories.unsqueeze(1) * goals[idx]).sum(2) / (normed_trajectories * torch.norm(goals[idx]))
        goal_similarities = goal_similarities.reshape(goal_similarities.shape[0])
        sorted_indices = goal_similarities.argsort(descending=True).tolist()
        sorted_fxfs = [f'({x // fact_map_len}, {x % fact_map_len})' for x in sorted_indices]

        correct_indices = tuple(fact_strings.index(x.normalized_value) for x in graph.premises)
        correct_indices = f'({correct_indices[0]}, {correct_indices[1]})'
        rank = sorted_fxfs.index(correct_indices)
        ranks.append(rank)

    mrr = sum([1 / (x+1) for x in ranks]) / max(1, len(ranks))
    return mrr

if __name__ == "__main__":
    DATA = 'eb'

    if DATA == 'eb':
        data = ENTAILMENT_BANK_FOLDER / 'task_1/test.jsonl'

        graphs = create_shallow_graphs(
            data,
            depth=1,
            min_depth=1,
            max_depth=1
        )

    elif DATA == 'babi':
        from multi_type_search.scripts.experiment_babi_task15_step_selection import build_examples
        graphs = build_examples(-1)
        for graph in graphs:
            graph.distractor_premises = []
    else:
        raise Exception(f"Unknown data type {DATA}")

    graphs = [x for x in graphs if len(x.premises) == 2]

    # graphs = graphs[0:10]

    WORD2VEC_MAP_FILE = Path('w2v.pkl')
    GLOVE_MAP_FILE = Path('glove.pkl')

    BUILD_W2V = False
    BUILD_GLOVE = False

    RUN_W2V = False
    RUN_GLOVE = False

    w2v_node_embedder = NodeEmbedder('word2vec_contrastive_model')

    if BUILD_W2V:
        w2v_fact_map, w2v_goal_map = build_embedding_table(deepcopy(graphs), w2v_node_embedder)

        with WORD2VEC_MAP_FILE.open('wb') as f:
            pickle.dump({'facts': w2v_fact_map, 'goals': w2v_goal_map}, f)

    if RUN_W2V:
        with WORD2VEC_MAP_FILE.open('rb') as f:
            w2v_map = pickle.load(f)
            w2v_fact_map = w2v_map['facts']
            w2v_goal_map = w2v_map['goals']

        w2v_mrr = t3_step_recall_with_map(
            w2v_fact_map,
            w2v_goal_map,
            graphs
        )

        print(f"W2V Mrr = {w2v_mrr:.8f}")

    glove_node_embedder = NodeEmbedder('glove_contrastive_model')

    if BUILD_GLOVE:
        glove_fact_map, glove_goal_map = build_embedding_table(deepcopy(graphs), glove_node_embedder)

        with GLOVE_MAP_FILE.open('wb') as f:
            pickle.dump({'facts': glove_fact_map, 'goals': glove_goal_map}, f)

    if RUN_GLOVE:
        with GLOVE_MAP_FILE.open('rb') as f:
            glove_map = pickle.load(f)
            glove_fact_map = glove_map['facts']
            glove_goal_map = glove_map['goals']

        glove_mrr = t3_step_recall_with_map(
            glove_fact_map,
            glove_goal_map,
            graphs
        )

        print(f'GloVe Mrr = {glove_mrr:.8f}')

    # === ---- === #

    DATA = 'babi'

    if DATA == 'eb':
        data = ENTAILMENT_BANK_FOLDER / 'task_1/test.jsonl'

        graphs = create_shallow_graphs(
            data,
            depth=1,
            min_depth=1,
            max_depth=1
        )

    elif DATA == 'babi':
        from multi_type_search.scripts.experiment_babi_task15_step_selection import build_examples
        graphs = build_examples(-1)
        for graph in graphs:
            graph.distractor_premises = []
    else:
        raise Exception(f"Unknown data type {DATA}")

    graphs = [x for x in graphs if len(x.premises) == 2]

    # graphs = graphs[0:10]

    WORD2VEC_MAP_FILE = Path('w2v_babi.pkl')
    GLOVE_MAP_FILE = Path('glove_babi.pkl')

    BUILD_W2V = True
    BUILD_GLOVE = True

    RUN_W2V = True
    RUN_GLOVE = True

    w2v_node_embedder = NodeEmbedder('word2vec_contrastive_model')

    if BUILD_W2V:
        w2v_fact_map, w2v_goal_map = build_embedding_table(deepcopy(graphs), w2v_node_embedder)

        with WORD2VEC_MAP_FILE.open('wb') as f:
            pickle.dump({'facts': w2v_fact_map, 'goals': w2v_goal_map}, f)

    if RUN_W2V:
        with WORD2VEC_MAP_FILE.open('rb') as f:
            w2v_map = pickle.load(f)
            w2v_fact_map = w2v_map['facts']
            w2v_goal_map = w2v_map['goals']

        w2v_mrr = t3_step_recall_with_map(
            w2v_fact_map,
            w2v_goal_map,
            graphs
        )

        print(f"W2V Mrr = {w2v_mrr:.8f}")

    glove_node_embedder = NodeEmbedder('glove_contrastive_model')

    if BUILD_GLOVE:
        glove_fact_map, glove_goal_map = build_embedding_table(deepcopy(graphs), glove_node_embedder)

        with GLOVE_MAP_FILE.open('wb') as f:
            pickle.dump({'facts': glove_fact_map, 'goals': glove_goal_map}, f)

    if RUN_GLOVE:
        with GLOVE_MAP_FILE.open('rb') as f:
            glove_map = pickle.load(f)
            glove_fact_map = glove_map['facts']
            glove_goal_map = glove_map['goals']

        glove_mrr = t3_step_recall_with_map(
            glove_fact_map,
            glove_goal_map,
            graphs
        )

        print(f'GloVe Mrr = {glove_mrr:.8f}')