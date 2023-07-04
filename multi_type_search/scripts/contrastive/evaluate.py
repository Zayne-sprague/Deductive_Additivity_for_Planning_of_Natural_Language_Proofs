import torch
import math
import faiss
from tqdm import tqdm
from typing import List, Dict
import numpy as np

from multi_type_search.search.search_model.types.contrastive.contrastive_utils import cosine_similarity_metric


def faiss_iterative_search(
        embeddings: torch.Tensor,
        targets: torch.Tensor,
        top_k: int = 10,
        iterations: int = 2,
        batch_size: int = 16,
        use_abs: bool = False,
        gamma: float = 10.
):
    embs = embeddings.cpu().numpy()
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(embeddings.shape[-1])
    index.add(embs)
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)

    sets = []

    for batched_idx in tqdm(range(0, targets.shape[0], batch_size), desc='Searching', total=math.ceil(targets.shape[0] / batch_size)):
        batch_targets = targets[batched_idx:min([batched_idx + batch_size, targets.shape[0]])].cpu().numpy()
        faiss.normalize_L2(batch_targets)
        batch_size = len(batch_targets)

        scores = np.zeros([batch_size])
        support_sets = np.array([])

        for hop in range(iterations):
            if hop == 0:
                D, I = index.search(batch_targets, len(embeddings) if use_abs else top_k)
                if use_abs:
                    D, I = __faiss_get_abs_topk__(D, I, top_k)
                scores = D
                batch_targets = ((gamma * np.expand_dims(batch_targets, axis=1)) - embs[I]).reshape(-1, embeddings.shape[-1])
                faiss.normalize_L2(batch_targets)
                support_sets = I.reshape([batch_size, top_k, 1])
            else:
                D, I = index.search(batch_targets, len(embeddings) if use_abs else top_k)
                if use_abs:
                    D, I = __faiss_get_abs_topk__(D, I, top_k)
                I = I.reshape(batch_size, -1)
                scores = (scores.reshape(-1, 1) + D).reshape(batch_size, -1)
                I_inds = scores.argsort()[:, ::-1][:, 0:top_k]
                scores = scores[np.arange(batch_size)[:, None], I_inds]
                sup_inds = np.floor(I_inds / top_k).astype('int')
                emb_inds = I[np.arange(batch_size)[:, None], I_inds]
                support_sets = np.concatenate([support_sets[np.arange(batch_size)[:, None], sup_inds], emb_inds.reshape([batch_size, top_k, 1])], axis=-1)
                batch_targets = ((gamma * batch_targets.reshape([batch_size, top_k, embeddings.shape[-1]])) - embs[emb_inds]).reshape(-1, embeddings.shape[-1])
                faiss.normalize_L2(batch_targets)

        sets.extend(support_sets.tolist())
    return sets

def __faiss_get_abs_topk__(D, I, top_k=10):
    D = abs(D)
    sorted_indx = D.argsort(-1)[:, ::-1]
    sorted_D = D[np.arange(len(D))[:, None], sorted_indx][:, :top_k]
    sorted_I = I[np.arange(len(D))[:, None], sorted_indx][:, :top_k]
    return sorted_D, sorted_I


def iterative_search__both(
        embeddings: torch.Tensor,
        target: torch.Tensor,
        top_k: int = 10,
        iterations: int = 2,
        gamma: float = 10.
):

    scores = cosine_similarity_metric(embeddings, target)
    top_embs = scores.argsort(descending=True)[0:top_k].detach().tolist()

    support_sets = [([x], scores[x].detach().item()) for x in top_embs]

    iteration = 1
    while iteration < iterations:
        candidates = []

        for (emb_indices, score) in support_sets:
            enc = embeddings[emb_indices].sum(dim=0)
            # enc = sum([embeddings[x] for x in emb_indices])
            trajectory_scores = cosine_similarity_metric(enc + embeddings, target)
            trajectory_scores += cosine_similarity_metric(embeddings, (gamma * target) - enc)

            ranked_trajectories = trajectory_scores.argsort(descending=True).detach().tolist()

            top_embs = []
            for top_emb in ranked_trajectories:
                # if top_emb in emb_indices:
                #     continue

                top_embs.append(top_emb)
                if len(top_embs) == top_k:
                    break

            candidates.extend([([*emb_indices, x], score + trajectory_scores[x].detach().item()) for x in top_embs])

        support_sets = list(sorted(candidates, key=lambda tup:tup[1], reverse=True))[:top_k]
        iteration += 1

    return support_sets

def iterative_search(
        embeddings: torch.Tensor,
        target: torch.Tensor,
        top_k: int = 10,
        iterations: int = 2,
        exclusion_list: List[int] = ()
):

    scores = cosine_similarity_metric(embeddings, target)
    raw_top_embs = scores.argsort(descending=True)[0:top_k+len(exclusion_list)].detach().tolist()

    top_embs = []

    if len(exclusion_list) > 0:
        for idx in raw_top_embs:
            if idx not in exclusion_list:
                top_embs.append(idx)
    else:
        top_embs = raw_top_embs

    support_sets = [([x], scores[x].detach().item()) for x in top_embs]

    iteration = 1
    while iteration < iterations:
        candidates = []

        for (emb_indices, score) in support_sets:
            enc = embeddings[emb_indices].sum(dim=0)
            # enc = sum([embeddings[x] for x in emb_indices])
            trajectory_scores = cosine_similarity_metric(enc + embeddings, target)
            ranked_trajectories = trajectory_scores.argsort(descending=True).detach().tolist()

            top_embs = []
            for top_emb in ranked_trajectories:
                if top_emb in emb_indices or top_emb in exclusion_list:
                    continue

                top_embs.append(top_emb)
                if len(top_embs) == top_k:
                    break

            candidates.extend([([*emb_indices, x], score + trajectory_scores[x].detach().item()) for x in top_embs])

        support_sets = list(sorted(candidates, key=lambda tup:tup[1], reverse=True))[:top_k]
        iteration += 1

    return support_sets


def iterative_search_mlp_head(
        model,
        embeddings: torch.Tensor,
        target: torch.Tensor,
        top_k: int = 10,
        iterations: int = 2,
):

    scores = cosine_similarity_metric(embeddings, target)
    top_embs = scores.argsort(descending=True)[0:top_k].detach().tolist()

    support_sets = [([x], scores[x].detach().item()) for x in top_embs]

    model.cuda().half()

    iteration = 1
    while iteration < iterations:
        candidates = []

        for (emb_indices, score) in support_sets:
            enc = embeddings[emb_indices].sum(dim=0)
            # enc = sum([embeddings[x] for x in emb_indices])
            # trajectory_scores = cosine_similarity_metric(enc + embeddings, target)
            pairs = torch.stack([enc.repeat(embeddings.shape[0], 1), embeddings], 1)
            pair_embs = torch.tensor([]).cuda()
            batch_size = 200

            for i in range(0, int(pairs.shape[0] / batch_size)+1):
                pair_embs = torch.concat([pair_embs, model.encode_pair(pairs[i*batch_size:min((i+1)*batch_size, pairs.shape[0])].cuda().half())], 0)

            trajectory_scores = cosine_similarity_metric(pair_embs, target, 1)

            ranked_trajectories = trajectory_scores.argsort(descending=True).detach().tolist()

            top_embs = []
            for top_emb in ranked_trajectories:
                # if top_emb in emb_indices:
                #     continue

                top_embs.append(top_emb)
                if len(top_embs) == top_k:
                    break

            candidates.extend([([*emb_indices, x], score + trajectory_scores[x].detach().item()) for x in top_embs])

        support_sets = list(sorted(candidates, key=lambda tup:tup[1], reverse=True))[:top_k]
        iteration += 1

    return support_sets


def iterative_search__subt(
        embeddings: torch.Tensor,
        target: torch.Tensor,
        top_k: int = 10,
        iterations: int = 2,
        gamma: float = 10.
):

    scores = cosine_similarity_metric(embeddings, target)
    top_embs = scores.argsort(descending=True)[0:top_k].detach().tolist()

    support_sets = [([x], scores[x].detach().item()) for x in top_embs]

    iteration = 1
    while iteration < iterations:
        candidates = []

        for (emb_indices, score) in support_sets:
            enc = embeddings[emb_indices].sum(dim=0)
            trajectory_scores = cosine_similarity_metric(embeddings, (gamma * target) - enc)
            ranked_trajectories = trajectory_scores.argsort(descending=True).detach().tolist()

            top_embs = []
            for top_emb in ranked_trajectories:
                if top_emb in emb_indices:
                    continue

                top_embs.append(top_emb)
                if len(top_embs) == top_k:
                    break

            candidates.extend([([*emb_indices, x], score + trajectory_scores[x].detach().item()) for x in top_embs])

        support_sets = list(sorted(candidates, key=lambda tup:tup[1], reverse=True))[:top_k]
        iteration += 1

    return support_sets


def iterative_search__mult(
        embeddings: torch.Tensor,
        target: torch.Tensor,
        top_k: int = 10,
        iterations: int = 2,
):

    scores = cosine_similarity_metric(embeddings, target)
    top_embs = scores.argsort(descending=True)[0:top_k].detach().tolist()

    support_sets = [([x], scores[x].detach().item()) for x in top_embs]

    iteration = 1
    while iteration < iterations:
        candidates = []

        for (emb_indices, score) in support_sets:
            enc = embeddings[emb_indices].sum(dim=0)
            trajectory_scores = cosine_similarity_metric(embeddings * enc, target)
            ranked_trajectories = trajectory_scores.argsort(descending=True).detach().tolist()

            top_embs = []
            for top_emb in ranked_trajectories:
                if top_emb in emb_indices:
                    continue

                top_embs.append(top_emb)
                if len(top_embs) == top_k:
                    break

            candidates.extend([([*emb_indices, x], score + trajectory_scores[x].detach().item()) for x in top_embs])

        support_sets = list(sorted(candidates, key=lambda tup:tup[1], reverse=True))[:top_k]
        iteration += 1

    return support_sets

def iterative_search__subtractive(
        embeddings: torch.Tensor,
        target: torch.Tensor,
        top_k: int = 10,
        iterations: int = 2,
):

    scores = cosine_similarity_metric(embeddings, target)
    top_embs = scores.argsort(descending=True)[0:top_k].detach().tolist()

    support_sets = [([x], scores[x].detach().item()) for x in top_embs]

    iteration = 1
    while iteration < iterations:
        candidates = []

        for (emb_indices, score) in support_sets:
            enc = embeddings[emb_indices].sum(dim=0)
            trajectory_scores = cosine_similarity_metric(embeddings - enc, target)
            ranked_trajectories = trajectory_scores.argsort(descending=True).detach().tolist()

            top_embs = []
            for top_emb in ranked_trajectories:
                if top_emb in emb_indices:
                    continue

                top_embs.append(top_emb)
                if len(top_embs) == top_k:
                    break

            candidates.extend([([*emb_indices, x], score + trajectory_scores[x].detach().item()) for x in top_embs])

        support_sets = list(sorted(candidates, key=lambda tup:tup[1], reverse=True))[:top_k]
        iteration += 1

    return support_sets


def iterative_search__max_pool(
        embeddings: torch.Tensor,
        target: torch.Tensor,
        top_k: int = 10,
        iterations: int = 2,
):

    scores = cosine_similarity_metric(embeddings, target)
    top_embs = scores.argsort(descending=True)[0:top_k].detach().tolist()

    support_sets = [([x], scores[x].detach().item()) for x in top_embs]

    iteration = 1
    while iteration < iterations:
        candidates = []

        for (emb_indices, score) in support_sets:
            enc = embeddings[emb_indices].sum(dim=0)
            # trajectories, _ = torch.max(torch.stack([torch.cat([enc.unsqueeze(0), v.unsqueeze(0)], dim=0) for v in embeddings]), dim=1)
            trajectory_scores = cosine_similarity_metric(torch.max(enc.unsqueeze(0), embeddings), target)
            ranked_trajectories = trajectory_scores.argsort(descending=True).detach().tolist()

            top_embs = []
            for top_emb in ranked_trajectories:
                if top_emb in emb_indices:
                    continue

                top_embs.append(top_emb)
                if len(top_embs) == top_k:
                    break

            candidates.extend([([*emb_indices, x], score + trajectory_scores[x].detach().item()) for x in top_embs])

        support_sets = list(sorted(candidates, key=lambda tup:tup[1], reverse=True))[:top_k]
        iteration += 1

    return support_sets


def iterative_search__min_pool(
        embeddings: torch.Tensor,
        target: torch.Tensor,
        top_k: int = 10,
        iterations: int = 2,
):

    scores = cosine_similarity_metric(embeddings, target)
    top_embs = scores.argsort(descending=True)[0:top_k].detach().tolist()

    support_sets = [([x], scores[x].detach().item()) for x in top_embs]

    iteration = 1
    while iteration < iterations:
        candidates = []

        for (emb_indices, score) in support_sets:
            enc = embeddings[emb_indices].sum(dim=0)
            # trajectories, _ = torch.max(torch.stack([torch.cat([enc.unsqueeze(0), v.unsqueeze(0)], dim=0) for v in embeddings]), dim=1)
            trajectory_scores = cosine_similarity_metric(torch.min(enc.unsqueeze(0), embeddings), target)
            ranked_trajectories = trajectory_scores.argsort(descending=True).detach().tolist()

            top_embs = []
            for top_emb in ranked_trajectories:
                if top_emb in emb_indices:
                    continue

                top_embs.append(top_emb)
                if len(top_embs) == top_k:
                    break

            candidates.extend([([*emb_indices, x], score + trajectory_scores[x].detach().item()) for x in top_embs])

        support_sets = list(sorted(candidates, key=lambda tup:tup[1], reverse=True))[:top_k]
        iteration += 1

    return support_sets


def iterative_search__avg_pool(
        embeddings: torch.Tensor,
        target: torch.Tensor,
        top_k: int = 10,
        iterations: int = 2,
):

    scores = cosine_similarity_metric(embeddings, target)
    top_embs = scores.argsort(descending=True)[0:top_k].detach().tolist()

    support_sets = [([x], scores[x].detach().item()) for x in top_embs]

    iteration = 1
    while iteration < iterations:
        candidates = []

        for (emb_indices, score) in support_sets:
            enc = embeddings[emb_indices].sum(dim=0)
            # trajectories, _ = torch.max(torch.stack([torch.cat([enc.unsqueeze(0), v.unsqueeze(0)], dim=0) for v in embeddings]), dim=1)
            trajectory_scores = cosine_similarity_metric((enc + embeddings) / 2, target)
            ranked_trajectories = trajectory_scores.argsort(descending=True).detach().tolist()

            top_embs = []
            for top_emb in ranked_trajectories:
                if top_emb in emb_indices:
                    continue

                top_embs.append(top_emb)
                if len(top_embs) == top_k:
                    break

            candidates.extend([([*emb_indices, x], score + trajectory_scores[x].detach().item()) for x in top_embs])

        support_sets = list(sorted(candidates, key=lambda tup:tup[1], reverse=True))[:top_k]
        iteration += 1

    return support_sets


def rank_support_sets(
        gold_indices: List[int],
        support_sets: List[List[str]],
):
    metrics = []

    for set_idx, (support_set, _) in enumerate(support_sets):
        hits = 0
        gold_idx = []
        for idx in support_set:
            if idx in gold_indices and idx not in gold_idx:
                gold_idx.append(idx)
                hits += 1

        metrics.append({'support_set_idx': set_idx, 'hits': hits, 'percentage': hits / len(gold_indices)})

    metrics = list(sorted(metrics, key=lambda x: x['percentage'], reverse=True))

    return metrics


def multiple_support_set_metrics(
        all_metrics: List[List[Dict[str, any]]]
):
    atleast_one = 0
    all = 0
    total = len(all_metrics)

    for metric in all_metrics:
        top = metric[0]
        atleast_one += 1 if top['percentage'] > 0. else 0
        all += 0 if top['percentage'] < 1. else 1

    return {'all': all/total if total > 0 else 0 , 'atleast_one': atleast_one/total if total > 0 else 0}


def gold_index_evaluation(embeddings, target, gold_args):
    def hop(enc):
        scores = cosine_similarity_metric(embeddings, target + enc)
        scores += cosine_similarity_metric(embeddings, target - enc).abs()
        return scores.argsort(
            descending=True).detach().tolist()

    scores = cosine_similarity_metric(embeddings, target)
    top_embs = scores.argsort(descending=True).detach().tolist()

    single_hop_arg1 = top_embs.index(gold_args[0])
    single_hop_arg2 = top_embs.index(gold_args[1])

    # trajectories_arg1 = cosine_similarity_metric(embeddings, target - embeddings[gold_args[0]]).argsort(descending=True).detach().tolist()
    # trajectories_arg2 = cosine_similarity_metric(embeddings, target - embeddings[gold_args[1]]).argsort(descending=True).detach().tolist()
    trajectories_arg1 = hop(embeddings[gold_args[0]])
    trajectories_arg2 = hop(embeddings[gold_args[1]])

    second_hop_arg1 = trajectories_arg2.index(gold_args[0])
    second_hop_arg2 = trajectories_arg1.index(gold_args[1])

    return {
        'single_hop_args': [single_hop_arg1, single_hop_arg2],
        'second_hop_args': [second_hop_arg1, second_hop_arg2]
    }