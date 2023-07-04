import numpy as np
import time
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader
from typing import Iterable, Callable, List, Union
from copy import deepcopy

try:
    import apex
    apex.amp.register_half_function(torch, 'einsum')
    from apex import amp
except:
    print("NO APEX")

from pathlib import Path
import shutil
from plotly.tools import mpl_to_plotly
from statistics import stdev

from multi_type_search.search.search_model.types.contrastive import ContrastiveModel, NonParametricVectorSpace, \
    RobertaEncoder, RobertaMomentumEncoder, RawGPT3Encoder, ProjectedGPT3Encoder
from multi_type_search.search.search_model import NodeEmbedder
from multi_type_search.search.search_model.types.contrastive.contrastive_utils import cosine_similarity_metric
from multi_type_search.scripts.contrastive.contrastive_losses import ContrastiveLoss
from multi_type_search.scripts.contrastive.contrastive_dataset import ContrastiveDataset
from multi_type_search.scripts.contrastive.corpus_dataset import CorpusDataset
from multi_type_search.scripts.contrastive.goals_dataset import GoalDataset
from multi_type_search.search.search_model.types.step_model import StepModel

# from multi_type_search.scripts.contrastive.evaluate import iterative_search, rank_support_sets, \
#     multiple_support_set_metrics, iterative_search__both, faiss_iterative_search, iterative_search__subt, \
#     iterative_search__mult, gold_index_evaluation, iterative_search_mlp_head, iterative_search__subtractive, \
#     iterative_search__max_pool, iterative_search__min_pool, iterative_search__avg_pool
from multi_type_search.scripts.heuristic_mrr_benchmark import mrr
from multi_type_search.search.step_selector import VectorSpaceSelector


# from multi_type_search.retrieval.scripts.vector_space_analysis import plot_mhop_add, plot_mhop_mlp_head
tdevice = 'cuda' if torch.cuda.is_available() else 'cpu'


def condition_number_regularization(model):
    ni = 0
    norm_loss = 0.
    for param in model.parameters(recurse=True):
        if len(param.shape) == 1:
            continue

        ni += 1
        norm_loss += param.norm().half() * torch.linalg.pinv(param).norm().half()

    norm_loss /= ni
    return norm_loss


def create_trajectories(x, y, method: str = 'add'):
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


def handle_batch(
    model,
    batch,
    momentum_model: bool = False,
    trajectory_creation_method: str = 'add',
    graph_goal: bool = False
):
    _, _, args, deductive_goal, ggoal, _, _, _, ids = batch

    args = move_to_cuda(args)
    deductive_goal = move_to_cuda(deductive_goal)
    ggoal_embs = None

    if graph_goal:
        ggoal = move_to_cuda(ggoal)

    if momentum_model:
        arg_embs = model(args)
        deductive_goal_embs = model(deductive_goal)

        if graph_goal:
            ggoal_embs = model(ggoal)
    else:
        arg_embs, _, _ = model(args)
        deductive_goal_embs, _, _ = model(deductive_goal)
        if graph_goal:
            ggoal_embs, _, _ = model(ggoal)

    arg_embs = arg_embs.view([deductive_goal_embs.shape[0], 2, deductive_goal_embs.shape[-1]])
    arg1 = arg_embs[:, 0, :]
    arg2 = arg_embs[:, 1, :]
    trajectories = create_trajectories(arg1, arg2, trajectory_creation_method)

    return trajectories, deductive_goal_embs, ggoal_embs, arg1, arg2, ids


def handle_batch_with_momentum(
    model: RobertaMomentumEncoder,
    batch,
    momentum_update: float = 0.999,
    use_memory_bank: bool = False,
    trajectory_creation_method: str = 'add'
):
    if not use_memory_bank:
        model.momentum_update(momentum=momentum_update)

    _, _, args, deductive_goal, _, _, _, ids = batch

    args = move_to_cuda(args)
    deductive_goal = move_to_cuda(deductive_goal)

    if use_memory_bank:
        arg_embs = model.q_enc(args)
    else:
        arg_embs = model.k_enc(args)

    new_deductive_goal_embs = model.q_enc(deductive_goal)
    deductive_goal_embs = torch.cat([new_deductive_goal_embs, model.query_queue.clone().detach()], axis=0)

    arg_embs = arg_embs.view([-1, 2, deductive_goal_embs.shape[-1]])
    arg1 = torch.cat([arg_embs[:, 0, :], model.arg1_queue.clone().detach()], axis=0)
    arg2 = torch.cat([arg_embs[:, 1, :], model.arg2_queue.clone().detach()], axis=0)


    trajectories = create_trajectories(arg1, arg2, trajectory_creation_method)

    model.dequeue_and_enqueue(arg_embs[:, 0, :], 'arg1_queue')
    model.dequeue_and_enqueue(arg_embs[:, 1, :], 'arg2_queue')

    # new_trajectories = model.encode_pair(arg_embs)
    # trajectories = torch.cat([new_trajectories, model.arg1_queue.clone().detach()], axis=0)

    # model.dequeue_and_enqueue(new_trajectories, 'arg1_queue')
    model.dequeue_and_enqueue(new_deductive_goal_embs, 'query_queue')

    return trajectories, deductive_goal_embs, None, arg1, arg2, ids


# def to_cuda(x):
#     """Helper """
#     if torch.cuda.is_available():
#         if isinstance(x, object) and not isinstance(x, dict):
#             x.data['input_ids'] = x.data['input_ids'].cuda()
#             x.data['attention_mask'] = x.data['attention_mask'].cuda()
#             return x
#         if isinstance(x, dict):
#             return {k: v.cuda() for k, v in x.items()}
#         return x.cuda()
#     return x

def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.to(tdevice)
        elif isinstance(maybe_tensor, dict):
            return {
                key: _move_to_cuda(value)
                for key, value in maybe_tensor.items()
            }
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(sample)

def train_epoch(
        model: Union[NonParametricVectorSpace, RobertaMomentumEncoder, RobertaEncoder],
        dataloader: DataLoader,
        optimizer: Optimizer,
        loss_fn: ContrastiveLoss,
        deductive_loss_term_weight: float = 1.0,
        graph_goal_loss_term_weight: float = 0.0,
        arg_goal_sim_loss_term_weight: float = 0.0,
        subt_loss_term_weight: float = 0.0,
        subt_gamma_term: float = 10.,
        condition_number_regularization_term_weight: float = 0.0,
        mse_term_weight: float = 0.0,
        cosinesim_term_weight: float = 0.0,
        batch_callback: Callable = None,
        scaler=None,
        use_momentum: bool = False,
        use_memory_bank: bool = False,
        momentum_update: float = 0.999,
        trajectory_creation_method: str = 'add',
        hf_model: bool = False

):
    model.train()

    total_loss = 0.
    total_batches = len(dataloader)

    for idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        if use_momentum or use_memory_bank:
            trajectories, deductive_goal_embs, graph_goal_embs, arg1, arg2, ids = handle_batch_with_momentum(model, batch, momentum_update, use_memory_bank, trajectory_creation_method=trajectory_creation_method)
        else:
            trajectories, deductive_goal_embs, graph_goal_embs, arg1, arg2, ids = handle_batch(model, batch, momentum_model=hf_model, trajectory_creation_method=trajectory_creation_method, graph_goal=graph_goal_loss_term_weight>0.0)

        positive_matrix = batch[0]

        #with torch.cuda.amp.autocast():
        loss_term = 0.
        if deductive_loss_term_weight > 0.:
            _score = loss_fn(trajectories, deductive_goal_embs, positive_matrix)
            loss_term = loss_term + deductive_loss_term_weight * _score
        if subt_loss_term_weight > 0.:
            loss_term = loss_term + subt_loss_term_weight * loss_fn(arg1, (subt_gamma_term * deductive_goal_embs) - arg2, positive_matrix)
            loss_term = loss_term +  subt_loss_term_weight * loss_fn(arg2, (subt_gamma_term * deductive_goal_embs) - arg1, positive_matrix)
        if graph_goal_loss_term_weight > 0.:
            loss_term = loss_term +  graph_goal_loss_term_weight * loss_fn(trajectories, graph_goal_embs, positive_matrix)
        if arg_goal_sim_loss_term_weight > 0.:
            loss_term = loss_term +  arg_goal_sim_loss_term_weight * loss_fn(arg1, deductive_goal_embs, positive_matrix)
            loss_term = loss_term +  arg_goal_sim_loss_term_weight * loss_fn(arg2, deductive_goal_embs, positive_matrix)
        if mse_term_weight > 0.:
            loss_term = loss_term +  mse_term_weight * ((torch.nn.functional.normalize(deductive_goal_embs, dim=-1) - torch.nn.functional.normalize(trajectories, dim = -1))**2).sum(-1).mean()
        if cosinesim_term_weight > 0.:
            sims = cosine_similarity_metric(trajectories, deductive_goal_embs.unsqueeze(1))
            targs = torch.zeros_like(sims).to(tdevice)
            #targs = targs - 1.
            targs.fill_diagonal_(1.)
            cosine_loss = ((targs - sims) ** 2).mean()
            loss_term = loss_term +  cosinesim_term_weight * cosine_loss
        if condition_number_regularization_term_weight > 0.:
            loss_term = loss_term +  condition_number_regularization_term_weight * condition_number_regularization(model)

        #scaler.scale(loss_term).backward()
        #scaler.step(optimizer)
        #scaler.update()
        loss_term.backward()
        optimizer.step()

        loss_term = loss_term.item()
        total_loss += loss_term

        if batch_callback is not None:
            batch_callback(locals())

    total_loss = total_loss / total_batches

    return total_loss


def validation_callback(epoch, locals):
    global validation_batch, deductive_residual_data

    deductive_cosine_scores = locals.get("deductive_cosine_scores").detach()
    deductive_cosine_mean = locals.get("deductive_cosine_mean")
    mse_residuals = locals.get("mse_residuals").detach()
    mse_residual_mean = locals.get("mse_residual_mean")

    _, _, _, _, _, arg_strings, deductive_goal_strings, _ = locals.get('batch')

    deductive_residual_data.extend(list(zip(deductive_cosine_scores, mse_residuals, deductive_goal_strings, [(arg_strings[x], arg_strings[x+1]) for x in range(0, len(arg_strings), 2)])))

    wandb.log({
        'validation/mean_mse_score': mse_residual_mean,
        'validation/mean_cosine_score': deductive_cosine_mean,
        'epoch': epoch,
        'batch': validation_batch
    })
    validation_batch += 1

def validate_with_similarity(
        model: NonParametricVectorSpace,
        dataloader: DataLoader,
        batch_callback: Callable = None,
        using_momentum: bool = False,
        trajectory_creation_method: str = 'add',
        hf_model: bool = False
):
    model.eval()

    average_cosine_score = 0
    average_mse_score = 0
    total_batches = len(dataloader)

    for batch in dataloader:

        trajectories, deductive_goal_embs, _, _, _ = handle_batch(model, batch, momentum_model=using_momentum or hf_model, trajectory_creation_method=trajectory_creation_method)

        deductive_cosine_scores = cosine_similarity_metric(trajectories, deductive_goal_embs)
        deductive_cosine_mean = deductive_cosine_scores.mean().item()

        mse_residuals = ((deductive_goal_embs - trajectories)**2).sum(-1)
        mse_residual_mean = mse_residuals.mean()

        average_cosine_score += deductive_cosine_mean
        average_mse_score += mse_residual_mean

        if batch_callback is not None:
            batch_callback(locals())

    average_cosine_score /= total_batches
    average_mse_score /= total_batches
    return average_cosine_score, average_mse_score


def validation_with_iterative_recall(
    model: NonParametricVectorSpace,
    corpus_dataloader: DataLoader,
    goal_dataloader: DataLoader,
    search_types: List[str] = ('og_add'),
    top_k: int = 10,
    hops: int = 2,
    encode_corpus: bool = True,
    encode_queries: bool = True,
    encodings_file: Path = Path('./tmp_encodings.npy'),
    string_encodings_file: Path = Path('./tmp_strings.json'),
    batch_callback: Callable = None,
    iterative_search_callback: Callable = None,
    validation_size: int = None,
    using_momentum: bool = False,
    track_gold_indices: bool = False,
    subt_gamma: float = 10.,
    output_file: Path = None,
    plot_mhop: bool = True,
    use_wandb: bool = True,
    e:int = 1,
    hf_model: bool = False
):
    return {}
    # model.eval()
    #
    # time_to_encode_corpus = 0
    # time_to_encode_questions = 0
    #
    # if encode_corpus or encode_queries or not encodings_file.exists() or not string_encodings_file.exists():
    #     encodings_file.parent.mkdir(exist_ok=True, parents=True)
    #     string_encodings_file.parent.mkdir(exist_ok=True, parents=True)
    #
    #     if torch.cuda.is_available():
    #         arg_embs = torch.tensor([]).cuda()
    #         deductive_goal_embs = torch.tensor([]).cuda()
    #     else:
    #         arg_embs = torch.tensor([])
    #         deductive_goal_embs = torch.tensor([])
    #
    #     arg_strings = []
    #     deductive_goal_strings = []
    #     deductive_goal_args = []
    #
    #     model.half()
    #
    #     with torch.no_grad():
    #         if encode_corpus or not encodings_file.exists():
    #             time_to_encode_corpus = time.time()
    #             for batch in tqdm(corpus_dataloader, total=len(corpus_dataloader), desc='building corpus'):
    #                 toks, strings = batch
    #                 toks = move_to_cuda(toks)
    #                 if using_momentum or hf_model:
    #                     arg_embs = torch.cat((arg_embs, model(toks)), dim=0)
    #                 else:
    #                     arg_embs = torch.cat((arg_embs, model(toks)[0]), dim=0)
    #                 arg_strings.extend(strings)
    #             time_to_encode_corpus = time.time() - time_to_encode_corpus
    #
    #         if encode_queries or not string_encodings_file.exists():
    #             time_to_encode_questions = time.time()
    #             for batch in tqdm(goal_dataloader, total=len(goal_dataloader), desc='encoding targets'):
    #                 toks, str_args, str_goals = batch
    #                 toks = move_to_cuda(toks)
    #
    #                 if using_momentum or hf_model:
    #                     deductive_goal_embs = torch.cat((deductive_goal_embs, model(toks)), dim=0)
    #                 else:
    #                     deductive_goal_embs = torch.cat((deductive_goal_embs, model(toks)[0]), dim=0)
    #                 deductive_goal_strings.extend(str_goals)
    #
    #                 deductive_goal_args.extend([[arg_strings.index(y) for y in x] for x in str_args])
    #             time_to_encode_questions = time.time() - time_to_encode_questions
    #
    #     del toks
    #
    #     with encodings_file.open('wb') as f:
    #         np.save(f, arg_embs.detach().cpu().numpy())
    #         np.save(f, deductive_goal_embs.detach().cpu().numpy())
    #     with string_encodings_file.open('w') as f:
    #         json.dump({'corpus': arg_strings, 'deductive_goal_strings': deductive_goal_strings, 'deductive_goal_gold_args': deductive_goal_args}, f)
    #
    # with open(str(encodings_file), 'rb') as f:
    #     arg_embs = torch.tensor(np.load(f))
    #     deductive_goal_embs = torch.tensor(np.load(f))
    # with string_encodings_file.open('rb') as f:
    #     data = json.load(f)
    #     arg_strings = data.get('corpus')
    #     deductive_goal_strings = data.get('deductive_goal_strings')
    #     deductive_goal_args = data.get('deductive_goal_gold_args')
    #
    # if encodings_file.name == 'tmp_encodings.npy':
    #     os.remove(str(encodings_file))
    # if string_encodings_file.name == 'tmp_strings.json':
    #     os.remove(str(string_encodings_file))
    #
    # val_metrics = {'time_to_encode_questions': time_to_encode_questions, 'time_to_encode_corpus': time_to_encode_corpus}
    #
    # if validation_size:
    #     deductive_goal_embs = deductive_goal_embs[0:min(validation_size, len(deductive_goal_embs))]
    #     deductive_goal_args = deductive_goal_args[0:min(validation_size, len(deductive_goal_args))]
    # else:
    #     validation_size = len(deductive_goal_embs)
    #
    # # free up some space
    # model = model.cpu()
    #
    # if plot_mhop:
    #     # plt = plot_mhop_mlp_head(model, deductive_goal_embs, arg_embs, deductive_goal_args)
    #     arg_embs = arg_embs.to(tdevice)
    #     deductive_goal_embs = deductive_goal_embs.to(tdevice)
    #     plt = plot_mhop_add(deductive_goal_embs, arg_embs, deductive_goal_args)
    #     if use_wandb:
    #         from plotly.tools import mpl_to_plotly
    #
    #         wandb.log({f"mhop_sims": wandb.Image(plt)})
    #     else:
    #         plt.show()
    #
    # if track_gold_indices:
    #     single_hop_total = 0
    #     second_hop_total = 0
    #     secs = []
    #     for idx, (target, gold_args) in tqdm(enumerate(zip(deductive_goal_embs, deductive_goal_args)), desc='Finding gold indices...', total=validation_size):
    #         gold_metrics = gold_index_evaluation(embeddings=arg_embs, target=target, gold_args=gold_args)
    #         single_hop_total += sum(gold_metrics['single_hop_args'])
    #         second_hop_total += sum(gold_metrics['second_hop_args'])
    #         secs.extend(gold_metrics['second_hop_args'])
    #     val_metrics['gold_index_eval'] = {}
    #     val_metrics['gold_index_eval']['single_hop_average_index'] = int(single_hop_total / (validation_size * 2))
    #     val_metrics['gold_index_eval']['second_hop_average_index'] = int(second_hop_total / (validation_size * 2))
    #
    #
    # for eval_name in search_types:
    #     time_for_eval = time.time()
    #
    #     if eval_name == 'faiss':
    #         arg_embs = arg_embs.cpu()
    #         deductive_goal_embs = deductive_goal_embs.cpu()
    #         support_sets = faiss_iterative_search(arg_embs, deductive_goal_embs, top_k=top_k, iterations=hops, use_abs=False, gamma=subt_gamma)
    #         support_sets = [[[y, None] for y in x] for x in support_sets] # stupid formatting to match old school structure
    #     else:
    #         arg_embs = arg_embs.to(tdevice)
    #         deductive_goal_embs = deductive_goal_embs.to(tdevice)
    #
    #     all_support_sets = []
    #     all_set_metrics = []
    #     iterator = enumerate(zip(deductive_goal_embs, deductive_goal_args))
    #     if eval_name != 'faiss':
    #         iterator = tqdm(iterator, desc='Searching...', total=validation_size)
    #
    #     for idx, (target, gold_args) in iterator:
    #         if eval_name == 'og_add':
    #             goal_string = deductive_goal_strings[idx]
    #             excl = []
    #             if goal_string in arg_strings:
    #                 excl = [arg_strings.index(goal_string)]
    #             sets = iterative_search(arg_embs, target, top_k=top_k, iterations=hops, exclusion_list=excl)
    #         elif eval_name == 'og_both':
    #             sets = iterative_search__both(arg_embs, target, top_k=top_k, iterations=hops, gamma=subt_gamma)
    #         elif eval_name == 'og_subt':
    #             sets = iterative_search__subt(arg_embs, target, top_k=top_k, iterations=hops, gamma=subt_gamma)
    #         elif eval_name == 'mult':
    #             sets = iterative_search__mult(arg_embs, target, top_k=top_k, iterations=hops)
    #         elif eval_name == 'subtractive':
    #             sets = iterative_search__subtractive(arg_embs, target, top_k=top_k, iterations=hops)
    #         elif eval_name == 'max_pool':
    #             sets = iterative_search__max_pool(arg_embs, target, top_k=top_k, iterations=hops)
    #         elif eval_name == 'min_pool':
    #             sets = iterative_search__min_pool(arg_embs, target, top_k=top_k, iterations=hops)
    #         elif eval_name == 'avg_pool':
    #             sets = iterative_search__avg_pool(arg_embs, target, top_k=top_k, iterations=hops)
    #         elif eval_name == 'mlp_head':
    #             model.to(tdevice)
    #             sets = iterative_search_mlp_head(model, arg_embs, target, top_k=top_k, iterations=hops)
    #         elif eval_name == 'faiss':
    #             sets = support_sets[idx]
    #         else:
    #             raise Exception(f'Unknown eval type named: {eval_name}')
    #
    #         if output_file:
    #             all_support_sets.append({
    #                 'question': deductive_goal_strings[idx],
    #                 'candidate_chains': [[{'text': arg_strings[y]} for y in s[0]] for s in sets],
    #                 'gold_args': [arg_strings[x] for x in gold_args]
    #             })
    #         set_metrics = rank_support_sets(gold_args, sets)
    #         all_set_metrics.append(set_metrics)
    #
    #         if iterative_search_callback is not None:
    #             iterative_search_callback(locals())
    #
    #     time_for_eval = time.time() - time_for_eval
    #
    #     val_metrics[eval_name] = multiple_support_set_metrics(all_set_metrics)
    #     val_metrics[eval_name]['time_to_eval'] = time_for_eval
    #
    # model = model.to(tdevice)
    # model.float()
    #
    # if output_file:
    #     output_file.parent.mkdir(exist_ok=True, parents=True)
    #
    #     with jsonlines.open(str(output_file), 'w') as f:
    #         f.write_all(all_support_sets)
    #
    # return val_metrics

def validate_with_mrr(
        test_graphs,
        model: ContrastiveModel
):
    node_embedder = NodeEmbedder()
    node_embedder.model = model
    step_selector = VectorSpaceSelector(node_embedder)
    metrics, ranks = mrr(test_graphs, step_selector)
    return metrics['intermediate_mrr'], metrics['graph_goal_mrr'], stdev(ranks['intermediate_mrr']), stdev(ranks['graph_goal_ranks'])


if __name__ == "__main__":
    from multi_type_search.scripts.contrastive.contrastive_losses import NTXENTLoss, CosineLoss, MOCOLoss
    from multi_type_search.utils.paths import DATA_FOLDER, TRAINED_MODELS_FOLDER
    from multi_type_search.search.graph import Graph

    from torch import optim
    from torch.nn import DataParallel

    import wandb
    from tqdm import tqdm
    import os
    import json
    from jsonlines import jsonlines
    from functools import partial
    from datetime import datetime
    from argparse import ArgumentParser
    import pprint

    parser = ArgumentParser()

    parser.add_argument('--debug', '-d', action='store_true', help='Turns a lot of debugging heuristics on')
    parser.add_argument(
        '--use_train_as_validation', '-utav', action='store_true',
        help='Debugging heuristic that uses training data as the validation data (validate on training data)'
    )

    parser.add_argument('--wandb', '-w', action='store_true', help='Use WANDB for recording the run')
    parser.add_argument('--personal_wandb', '-wp', action='store_true', help='Use your personal wandb instead of ut_nlp_deduce')

    parser.add_argument(
        '--training_file', '-tf', type=str, default=str(DATA_FOLDER / 'full/hotpot/fullwiki_train.json'),
        help='Path to the training file'
    )
    parser.add_argument(
        '--validation_file', '-vf', type=str, default=str(DATA_FOLDER / 'full/hotpot/fullwiki_val.json'),
        help='Path to the validation file'
    )

    parser.add_argument(
        '--max_train_size', '-mts', type=int, default=None,
        help='Maximum number of training graphs to use.  None means use all.'
    )
    parser.add_argument(
        '--max_validation_size', '-mvs', type=int, default=None,
        help='Maximum number of validation graphs to use.  None means use all.'
    )

    parser.add_argument('--train', '-t', action='store_true', help='Train a model.')
    parser.add_argument('--do_mrr_validation', '-dmv', action='store_true', help='Evaluate on the MRR benchmark.')

    parser.add_argument('--track_sim_scores', '-tss', action='store_true',
                        help='Track MSE and Cosine Sim scores on validation data.')

    parser.add_argument('--plot_mhop', '-pm', action='store_true', help='Plot golds vs all distributions (only for Add for now)')

    parser.add_argument(
        '--iterative_search_eval', '-ise', type=str, nargs='+', choices=['og_subt', 'og_add', 'og_both', 'faiss', 'mlp_head', 'mult', 'subtractive', 'max_pool', 'min_pool', 'avg_pool'],
        default=[],
        help='Iterative Search Evaluation, choose between None (no eval), og_subt, og_add, og_both, faiss.  The first'
             ' choice will be used for checkpointing the model if training.',
    )

    parser.add_argument(
        '--trajectory_creation_method', '-tcm', type=str, default='add',
        choices=['add', 'subtract', 'multiply', 'max_pool', 'min_pool', 'avg_pool'],
        help='The method used to combine the premise embeddings ("add" is default)'
    )

    parser.add_argument(
        '--hf_model_name', '-hmn', type=str, default='roberta-base',
        help='Hugging face model name to use if applicable to the run.'
    )

    parser.add_argument(
        '--gpt3_cached_embeddings_file', '-gcef', type=str,
        help='File that contains the cached gpt3 embeddings. (Pickle file)'
    )

    parser.add_argument(
        '--gpt3_cached_strings_file', '-gcsf', type=str,
        help='File that contains the cached gpt3 strings (used for embeddings, Json file).'
    )
    parser.add_argument(
        '--gpt3_projection_head_layer_num', '-gphln', type=int, default=3,
        help='Number of layers in the GPT3 Projection head.'
    )
    parser.add_argument(
        '--gpt3_projection_head_type', '-gpht', type=str, default='linear',
        help='Number of layers in the GPT3 Projection head.'
    )

    parser.add_argument(
        '--iterative_search_eval_top_k', '-isetk', type=int, default=10,
        help='Number of support sets to return (will always take the top K scoring support sets)'
    )
    parser.add_argument(
        '--iterative_search_eval_hops', '-iseh', type=int, default=2,
        help='Number hops to do per support set (also size of each individual support set)'
    )
    parser.add_argument('--ise_track_gold_indices', '-tgi', action='store_true',
                        help='Track the average index of the gold args of examples for single hop and two hops')
    parser.add_argument(
        '--max_iterative_search_val_size', '-misvs', type=int,
        help='How many searches should be done.  Max validation size controls the size of the corpus, this controls how'
             ' many queries we will make in the evaluation loop.'
    )
    parser.add_argument(
        '--iterative_search_encodings_file', '-isef', type=str, default='./tmp_encodings.npy',
        help='NPY file to store the encoded corpus and questions file.  If default used, will be deleted after run.'
    )
    parser.add_argument(
        '--iterative_search_string_encodings_file', '-issef', type=str, default='./tmp_strings.json',
        help='Json file to store the strings for the iterative search eval. If default used, will be deleted after run.'
    )
    parser.add_argument(
        '--iterative_search_eval_output_folder', '-iseof', type=str,
        help='Store support sets for the eval in a folder.  It will be populated with files e1.jsonl, ..., en.jsonl for'
             ' each epoch of training.'
    )
    parser.add_argument(
        '--encode_corpus', '-ec', action='store_true',
        help='When doing the iterative search evaluation, always re-encode the corpus (training does this by default) '
    )
    parser.add_argument(
        '--encode_queries', '-eq', action='store_true',
        help='When doing the iterative search evaluation, always re-encode the queries (training does this by default) '
    )

    parser.add_argument('--batch_size', '-bs', type=int, help='Size of batch in training', default=50)
    parser.add_argument('--predict_batch_size', '-pbs', type=int, help='Size of batch for prediction/eval', default=100)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-5, help='Learning rate for training')
    parser.add_argument('--tau', '-tau', type=float, default=0.05, help='temperature for nxentloss')

    parser.add_argument('--loss_fn', '-lf', type=str, default='NTXENTLoss',
                        choices=['NTXENTLoss', 'MOCOLoss', 'Cosine'], help='Loss for training')

    parser.add_argument(
        '--two_p_loss_weights', '-tplw', type=float, nargs='+', default=[1.0],
        help='Weight of the (g, p1 + p2) loss per epoch.  If the epoch number exceeds the size of the weights given '
             'then the last value will be used (i.e. [0, 0.1, 0.9, 1] at epoch 5 1 will be used for then on)'
    )
    parser.add_argument(
        '--one_p_loss_weights', '-oplw', type=float, nargs='+', default=[0.0],
        help='Weight of the (g, p) loss per epoch.  If the epoch number exceeds the size of the weights given then'
             ' the last value will be used (i.e. [0, 0.1, 0.9, 1] at epoch 5 1 will be used for then on)'
    )
    parser.add_argument(
        '--subt_p_loss_weights', '-splw', type=float, nargs='+', default=[0.0],
        help='Weight of the total term for  (p2, [gamma * g] - p2) loss per epoch.  If the epoch number exceeds the size of the weights given '
             'then the last value will be used (i.e. [0, 0.1, 0.9, 1] at epoch 5 1 will be used for then on)'
    )
    parser.add_argument(
        '--subt_gamma_terms', '-sgt', type=float, nargs='+', default=[10.0],
        help='Weight of the gamma in the (p2, [gamma * g] - p2) loss function. if the epoch number exceeds the size of the weights given '
             'then the last value will be used (i.e. [0, 0.1, 0.9, 1] at epoch 5 1 will be used for then on)'
    )
    parser.add_argument(
        '--mse_term_weights', '-mselw', type=float, nargs='+', default=[0.0],
        help='Weight of the mse(p1+p2, g) loss per epoch.  If the epoch number exceeds the size of the weights given '
             'then the last value will be used (i.e. [0, 0.1, 0.9, 1] at epoch 5 1 will be used for then on)'
    )
    parser.add_argument(
        '--cosinesim_term_weights', '-cslw', type=float, nargs='+', default=[0.0],
        help='Weight of the cosine_sim(p1+p2, g) loss per epoch.  If the epoch number exceeds the size of the weights given '
             'then the last value will be used (i.e. [0, 0.1, 0.9, 1] at epoch 5 1 will be used for then on)'
    )
    parser.add_argument(
        '--graph_goal_loss_term_weights', '-ggltw', type=float, nargs='+', default=[0.0],
        help='Weight of the (graph goal, p1 + p2) loss per epoch.  If the epoch number exceeds the size of the weights given '
             'then the last value will be used (i.e. [0, 0.1, 0.9, 1] at epoch 5 1 will be used for then on)'
    )

    parser.add_argument(
        '--run_name', '-r', type=str, default=f'contrastive_model_{datetime.now().strftime("%Y_%m_%d__%I_%M_%S_%p")}',
        help='Name that will be used in checkpointing and WANDB.'
    )

    parser.add_argument(
        '--load_model_name', '-lmn', type=str,
        help='Name of the model you want to load (None will be a newly initialized model)'
    )

    parser.add_argument(
        '--use_memory_bank', '-umb', action='store_true',
        help='Use the Roberta Momentum Encoder to store more negatives via a memory bank (momentum will not be used)'
    )
    parser.add_argument(
        '--use_momentum_encoder', '-ume', action='store_true',
        help='Use the Roberta Momentum Encoder to store more negatives using MOCO like models'
    )
    parser.add_argument(
        '--momentum_update', '-mu', type=float, default=0.999,
        help='momentum hyperparam for the momentum encoder'
    )
    parser.add_argument(
        '--momentum_queue_size', '-mqs', type=int, default=2048,
        help='How large the momentum queue should be'
    )

    parser.add_argument(
        '--deductive_step_model', '-dsm', type=str, help='Deductive Step Model to use',
    )
    parser.add_argument(
        '--deductive_step_model_sample_rate', '-dsmsr', type=int, default=5,
        help='Number of samples to generate with the step model.'
    )
    parser.add_argument(
        '--training_dataset_cache_name', '-tdcm', type=str,
        help='Cache for the training dataset'
    )
    parser.add_argument(
        '--use_training_data_cache', '-utdc', action='store_true',
        help='Read in data from the cache'
    )
    parser.add_argument(
        '--overwrite_training_cache', '-otc', action='store_true',
        help='Overwrite the current training dataset cache.'
    )

    args = parser.parse_args()

    deductive_step_model = args.deductive_step_model
    deductive_step_model_sample_rate = args.deductive_step_model_sample_rate

    training_dataset_cache_name = args.training_dataset_cache_name
    use_training_data_cache = args.use_training_data_cache
    overwrite_training_cache = args.overwrite_training_cache

    # DEBUG: Controls if pin memory / num workers in the dataloader
    DEBUG = args.debug

    # WANDB: Controls whether or not the data is sent to WANDB
    WANDB = args.wandb
    PERSONAL_WANDB = args.personal_wandb

    if PERSONAL_WANDB:
        wandb.init(project="contrastive_tests", mode='disabled' if not WANDB else None)
    else:
        wandb.init(project="contrastive-training", mode='disabled' if not WANDB else None, entity="ut_nlp_deduce")

    if WANDB:
        wandb.run.name = args.run_name
        wandb.run.save()

    # Local paths from the top level data directory to training and validation files
    training_file = Path(args.training_file)
    # training_file = DATA_FOLDER / 'full/hotpot/fullwiki_train.json'
    # training_file = DATA_FOLDER / 'full/hotpot/fixed_declaritive_fw_train.jsonl'

    validation_file = Path(args.validation_file)
    # validation_file = DATA_FOLDER / 'full/hotpot/fullwiki_val.json'
    # validation_file = DATA_FOLDER / 'full/hotpot/fixed_declaritive_fw_val.jsonl'
    # validation_file = DATA_FOLDER / 'full/entailmentbank/task_1/shallow_test.json'
    # validation_file = DATA_FOLDER / 'full/morals/shallow_moral100.json'


    # Maximum sizes of the number of Graphs loaded per dataset.  None means all of the graphs.
    MAX_TRAIN_SIZE = args.max_train_size
    MAX_VALIDATION_SIZE = args.max_validation_size

    # Debugging tool: Use the training graphs for the training dataset as the validation graphs for val dataset.
    USE_TRAINING_AS_VALIDATION = args.use_train_as_validation

    Train = args.train
    MSE_VAL = False and args.track_sim_scores
    ITERATIVE_VAL = False and len(args.iterative_search_eval) > 0
    MRR_VAL = args.do_mrr_validation
    trajectory_creation_method: str = args.trajectory_creation_method


    # MODEL PARAMETERS

    # Simcse names. These are in order of the best performance from previous experiments (may not be 100% accurate tho)
    simcse_names = [
        'princeton-nlp/sup-simcse-roberta-base',
        'princeton-nlp/sup-simcse-bert-base-uncased',
        'princeton-nlp/sup-simcse-roberta-large',
        'princeton-nlp/unsup-simcse-roberta-large',
        'princeton-nlp/sup-simcse-bert-large-uncased',
        'princeton-nlp/unsup-simcse-bert-large-uncased',
    ]
    simcse_name = simcse_names[0]

    # Set this to True if you want ot use the Roberta Base encoder instead of SIMCSE and T5 (MDR uses this)
    USE_ROBERTA_BASE = True

    # Set this to False if you want to use T5 as the backbone instead of SIMCSE
    USE_SIMCSE = False

    # handling which backbone to use
    USE_SIMCSE = USE_SIMCSE and not USE_ROBERTA_BASE
    if not USE_SIMCSE:
        simcse_name = None

    # T5 Arguments
    t5_model_name = 't5-base'
    t5_token_max_length = 256

    # CUSTOM LAYER PARAMETERS

    # Two forward layers are used, this is their hidden dim. Look at nonparametric_vs.py for the model definition.
    fw_hidden_size = 1024
    embedding_size = 512
    residual_connections = True

    # Use the embeddings produced by the backbone (no extra layers)
    backbone_only = True

    # Do not train the backone model (T5 or Simcse)
    freeze_backbone = False

    # TRAINING PARAMETERS
    batch_size = args.batch_size
    corpus_batch_size = args.predict_batch_size

    learning_rate = args.learning_rate
    epochs = 100

    # Stop training if the score doesn't get better after N epochs. (score defined in training loop)
    do_checkpointing = True
    early_stopping_after = 10
    model_checkpoint_name = args.run_name
    #model_checkpoint_name = ''

    # Use the Normalized Temperature Cross Entropy loss or the custom Cosine Sim loss
    nxt_tau = args.tau

    # LOSS TERM PARAMETERS (0 turns the loss term off)

    # Standard loss(p1 + p2, deductive_embedding) term
    deductive_loss_term_weights = args.two_p_loss_weights
    subt_loss_term_weights = args.subt_p_loss_weights
    subt_gamma_terms = args.subt_gamma_terms
    #deductive_loss_term_weights = [1.0]

    # Loss(p1 + p2, graph_goal_embedding) term.  Only useful in entailment bank or datasets with different tree goals
    # than deductive goals (multi-hop)
    graph_goal_loss_term_weights = args.graph_goal_loss_term_weights

    # sim(p_i, goal) loss term
    arg_goal_sim_loss_term_weights = args.one_p_loss_weights
    
    #arg_goal_sim_loss_term_weights = [0.0]

    mse_term_weights = args.mse_term_weights
    cosinesim_term_weights = args.cosinesim_term_weights

    # Regularization term Loss(layers_condition_number) -- attempts to reduce volatility of matrix vector multiplication
    condition_number_regularization_term_weights = [0.0]

    if args.use_momentum_encoder or args.use_memory_bank:
        roberta_model = RobertaEncoder(roberta_model_name=args.hf_model_name)
        model = RobertaMomentumEncoder(
            roberta_model,
            args.momentum_queue_size,
            not args.use_memory_bank
        )
    elif args.hf_model_name:
        if args.hf_model_name == 'raw_gpt3':
            model = RawGPT3Encoder(args.gpt3_cached_embeddings_file, args.gpt3_cached_strings_file, allow_api_access=False)
            model.activate_key()
        elif args.hf_model_name == 'projected_gpt3':
            model = ProjectedGPT3Encoder(args.gpt3_cached_embeddings_file, args.gpt3_cached_strings_file, projection_head_layer_num=args.gpt3_projection_head_layer_num, projection_head_type=args.gpt3_projection_head_type, allow_api_access=False)
            model.activate_key()
        else:
            model = RobertaEncoder(roberta_model_name=args.hf_model_name)
    else:
        model = NonParametricVectorSpace(
            simcse_name=simcse_name,
            t5_model_name=t5_model_name,
            roberta_base=USE_ROBERTA_BASE,
            t5_token_max_length=t5_token_max_length,
            fw_hidden_size=fw_hidden_size,
            embedding_size=embedding_size,
            residual_connections=residual_connections,
            backbone_only=backbone_only,
            freeze_backbone=freeze_backbone
        )

    optimizer = optim.Adam(model.parameters(recurse=True), lr=learning_rate)

    if args.load_model_name:
        model, _ = model.load(TRAINED_MODELS_FOLDER / f'{args.load_model_name}/best_checkpoint.pth', tdevice, optimizer)

    
    if args.loss_fn == 'NTXENTLoss':
        loss_fn = NTXENTLoss(
            temperature=nxt_tau
        )
    elif args.loss_fn == 'MOCOLoss':
        loss_fn = MOCOLoss()
    elif args.loss_fn == 'Cosine':
        loss_fn = CosineLoss()

    if not Train:
        MAX_TRAIN_SIZE = 1
        
    training_graphs = [Graph.from_json(x) for x in (json.load(training_file.open('r')) if training_file.name.endswith('.json') else list(jsonlines.Reader(training_file.open('r'))))[:MAX_TRAIN_SIZE]]
    # training_graphs.extend([Graph.from_json(x) for x in (json.load(training_file2.open('r')) if training_file2.name.endswith('.json') else list(jsonlines.Reader(training_file2.open('r'))))[:MAX_TRAIN_SIZE]])

    validation_graphs = [Graph.from_json(x) for x in (json.load(validation_file.open('r')) if validation_file.name.endswith('.json') else list(jsonlines.Reader(validation_file.open('r'))))[:MAX_VALIDATION_SIZE]]

    if deductive_step_model:
        deductive_model = StepModel(
            deductive_step_model,
            max_output_length=128,
            num_return_sequences=deductive_step_model_sample_rate,
            device='cuda:0' if torch.cuda.is_available() else 'cpu'
        )
    else:
       deductive_model = None

    training_dataset = ContrastiveDataset(tokenizer=model.roberta_tokenizer if 'gpt3' not in args.hf_model_name else None)
    validation_dataset = ContrastiveDataset(tokenizer=model.roberta_tokenizer if 'gpt3' not in args.hf_model_name else None)
    corpus_dataset = CorpusDataset(tokenizer=model.roberta_tokenizer if 'gpt3' not in args.hf_model_name else None)
    goal_dataset = GoalDataset(tokenizer=model.tokenize if 'gpt3' not in args.hf_model_name else None)

    training_dataset.populate_examples(
        training_graphs,
        progress_bar=True,
        write_cache_name=training_dataset_cache_name,
        read_cache_name=training_dataset_cache_name if not overwrite_training_cache else None,
        write_cache_if_not_exist=True,
        sampler= None if not deductive_model else deductive_model.sample
    )


    del deductive_model


    if USE_TRAINING_AS_VALIDATION:
        validation_dataset.populate_examples(training_graphs)
        corpus_dataset.populate_examples(training_graphs)
        goal_dataset.populate_examples(training_graphs)
    else:
        validation_dataset.populate_examples(validation_graphs)
        corpus_dataset.populate_examples(validation_graphs)
        goal_dataset.populate_examples(validation_graphs)

    wandb.config.update({
        'training_data': str(training_file),
        'validation_data': str(validation_file),
        'using_roberta': USE_ROBERTA_BASE,
        'simcse_name': simcse_name,
        't5_model_name': t5_model_name,
        't5_token_max_length': t5_token_max_length,
        'fw_hidden_size': fw_hidden_size,
        'embedding_size': embedding_size,
        'residual_connections': residual_connections,
        'backbone_only': backbone_only,
        'freeze_backbone': freeze_backbone,
        'learning_rate': learning_rate,
        'nxt_tau': nxt_tau,
        'loss': args.loss_fn,
        'deductive_loss_term_weights': deductive_loss_term_weights,
        'graph_goal_loss_term_weights': graph_goal_loss_term_weights,
        'arg_goal_sim_loss_term_weights': arg_goal_sim_loss_term_weights,
        'subt_loss_term_weights': subt_loss_term_weights,
        'subt_gamma_terms': subt_gamma_terms,
        'mse_term_weights': mse_term_weights,
        'cosinesim_term_weights': cosinesim_term_weights,
        'condition_number_regularization_term_weights': condition_number_regularization_term_weights,
        'model_checkpoint_name': model_checkpoint_name,
        'hops': args.iterative_search_eval_hops,
        'top_k': args.iterative_search_eval_top_k,
    })

    if torch.cuda.is_available():
        model = model.cuda()

    if torch.cuda.device_count() > 1 and not DEBUG:
        model = DataParallel(model)

    training_dataloader = DataLoader(
        training_dataset, batch_size, shuffle=True, collate_fn=training_dataset.collate_fn,
        pin_memory=not DEBUG, num_workers=1 if DEBUG else 12
    )

    validation_dataloader = DataLoader(
        validation_dataset, batch_size, shuffle=False, collate_fn=validation_dataset.collate_fn,
        pin_memory=DEBUG, num_workers=1 if DEBUG else 12
    )

    corpus_dataloader = DataLoader(
        corpus_dataset, corpus_batch_size, shuffle=False, collate_fn=corpus_dataset.collate_fn,
        pin_memory=True, num_workers=1 if DEBUG else 12
    )

    goal_dataloader = DataLoader(
        goal_dataset, corpus_batch_size, shuffle=True, collate_fn=goal_dataset.collate_fn,
        pin_memory=DEBUG, num_workers=1 if DEBUG else 12
    )

    training_batch = 0
    validation_batch = 0

    # Callbacks are used to keep track of data during training/validation loops.
    def training_callback(epoch, locals):
        global training_batch

        loss_term = locals.get('loss_term')
        wandb.log({
            "training/batch_loss": loss_term,
            'epoch': epoch,
            'batch': training_batch
        })
        training_batch += 1

        training_pbar.update(1)
        training_pbar.set_description(f'Training Loss: {loss_term:.4f}')

    deductive_residual_data = []
    deductive_iterative_retrieval_data = {}

    def validation_build_enc_callback(locals):
        sets = locals.get('sets')
        set_metrics = locals.get('set_metrics')
        gold_args = locals.get('gold_args')
        string_args = locals.get('arg_strings')
        deductive_goals = locals.get('deductive_goal_strings')
        idx = locals.get('idx')
        eval_name = locals.get('eval_name')

        if eval_name not in deductive_iterative_retrieval_data:
            deductive_iterative_retrieval_data[eval_name] = []

        deductive_iterative_retrieval_data[eval_name].append([
            set_metrics[0]['percentage'],
            [gold_args[0] in sets[set_metrics[0]['support_set_idx']][0], gold_args[1] in sets[set_metrics[0]['support_set_idx']][0]],
            [string_args[x] for x in sets[set_metrics[0]['support_set_idx']][0]],
            [string_args[x] for x in gold_args],
            deductive_goals[idx]
        ])


    def corpus_encoder_callback(locals):
        corpus_encoder_pbar.update(1)

    # wandb.watch(model)

    if torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    best_score = None
    epochs_since_last_best = 0

    # ACTUAL TRAINING LOOP
    for e in tqdm(range(epochs), desc='Training model', total=epochs, position=0, leave=False):
        training_pbar = tqdm(f'Training Epoch {e + 1}', position=1, leave=False, total=len(training_dataloader))
        corpus_encoder_pbar = tqdm(f'Corpus Encoder {e + 1}', position=3, leave=False, total=len(corpus_dataloader) + len(goal_dataloader))

        if Train:
            deductive_loss_term_weight = deductive_loss_term_weights[e] if e < len(deductive_loss_term_weights) else deductive_loss_term_weights[-1]
            graph_goal_loss_term_weight = graph_goal_loss_term_weights[e] if e < len(graph_goal_loss_term_weights) else graph_goal_loss_term_weights[-1]
            arg_goal_sim_loss_term_weight = arg_goal_sim_loss_term_weights[e] if e < len(arg_goal_sim_loss_term_weights) else arg_goal_sim_loss_term_weights[-1]
            subt_loss_term_weight = subt_loss_term_weights[e] if e < len(subt_loss_term_weights) else subt_loss_term_weights[-1]
            subt_gamma_term = subt_gamma_terms[e] if e < len(subt_gamma_terms) else subt_gamma_terms[-1]
            mse_term_weight = mse_term_weights[e] if e < len(mse_term_weights) else mse_term_weights[-1]
            cosinesim_term_weight = cosinesim_term_weights[e] if e < len(cosinesim_term_weights) else cosinesim_term_weights[-1]
            condition_number_regularization_term_weight = condition_number_regularization_term_weights[e] if e < len(condition_number_regularization_term_weights) else condition_number_regularization_term_weights[-1]

            # Train on the loss terms specified above
            total_loss = train_epoch(
                model=model,
                dataloader=training_dataloader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                deductive_loss_term_weight=deductive_loss_term_weight,
                graph_goal_loss_term_weight=graph_goal_loss_term_weight,
                arg_goal_sim_loss_term_weight=arg_goal_sim_loss_term_weight,
                subt_loss_term_weight=subt_loss_term_weight,
                subt_gamma_term=subt_gamma_term,
                mse_term_weight=mse_term_weight,
                cosinesim_term_weight=cosinesim_term_weight,
                condition_number_regularization_term_weight=condition_number_regularization_term_weight,
                batch_callback=partial(training_callback, e),
                scaler=scaler,
                use_momentum=args.use_momentum_encoder,
                use_memory_bank=args.use_memory_bank,
                momentum_update=args.momentum_update,
                trajectory_creation_method=trajectory_creation_method,
                hf_model=args.hf_model_name
            )

        with torch.no_grad():
            if MSE_VAL:
                # Check the MSE for 1 - cossim(p1+p2, g) then plot it. Also stores important visualization data for WANDB
                # that shows examples in table form of cossim(p1 + p2, g) scores
                average_cosine_score, average_mse_score = validate_with_similarity(
                    model=model,
                    dataloader=validation_dataloader,
                    batch_callback=partial(validation_callback, e),
                    using_momentum=args.use_momentum_encoder or args.use_memory_bank,
                    hf_model = args.hf_model_name,
                    trajectory_creation_method=trajectory_creation_method
                )

            if ITERATIVE_VAL:
                subt_gamma_term = subt_gamma_terms[e] if e < len(subt_gamma_terms) else subt_gamma_terms[-1]

                # Validation loop that does iterative recall experiment we run in our evaluation.
                val_metrics = validation_with_iterative_recall(
                    model=model,
                    corpus_dataloader=corpus_dataloader,
                    goal_dataloader=goal_dataloader,
                    encode_corpus=Train or args.encode_corpus,
                    encode_queries=Train or args.encode_queries,
                    encodings_file=Path(args.iterative_search_encodings_file),
                    string_encodings_file=Path(args.iterative_search_string_encodings_file),
                    search_types=args.iterative_search_eval,
                    top_k=args.iterative_search_eval_top_k,
                    hops=args.iterative_search_eval_hops,
                    validation_size=args.max_iterative_search_val_size,
                    iterative_search_callback=validation_build_enc_callback,
                    batch_callback=corpus_encoder_callback,
                    using_momentum=args.use_momentum_encoder or args.use_memory_bank,
                    track_gold_indices=args.ise_track_gold_indices,
                    subt_gamma=subt_gamma_term,
                    output_file=None,#Path(args.iterative_search_eval_output_folder) / f'e{e}.jsonl',
                    plot_mhop=args.plot_mhop,
                    use_wandb=WANDB,
                    e=e,
                    hf_model=args.hf_model_name
                )

            if MRR_VAL:
                mrr_val_metrics = validate_with_mrr(validation_graphs, model)


        if Train:
            epoch_data = {
                'training/total_loss': total_loss,
                'epoch': e,
                'batch': training_batch,
            }
        else:
            epoch_data = {}



        if MSE_VAL:
            # DATA VISUALIZATION WITH WANDB
            deductive_residual_data = list(sorted(deductive_residual_data, key=lambda x: x[0], reverse=True))

            # TODO - the worst line of code ever.
            # even_sampled_indices = list(set(sorted(np.round(np.linspace(0, len(deductive_residual_data) - 1, 20)).astype(int).tolist())))
            even_sampled_indices = list(range(0, len(deductive_residual_data)))

            similarity_score_table = wandb.Table(data=pd.DataFrame({
                'idx': [idx for idx in even_sampled_indices],
                'Cosine Sim': [deductive_residual_data[idx][0] for idx in even_sampled_indices],
                'MSE Score': [deductive_residual_data[idx][1] for idx in even_sampled_indices],
                'Target': [deductive_residual_data[idx][2] for idx in even_sampled_indices],
                'Gold Arg 1': [deductive_residual_data[idx][3][0] for idx in even_sampled_indices],
                'Gold Arg 2': [deductive_residual_data[idx][3][1] for idx in even_sampled_indices]
            }))

            epoch_data['similarity_scores'] = similarity_score_table
            epoch_data['validation/average_mse_score'] = average_mse_score
            epoch_data['validation/average_cosine_score'] = average_cosine_score


        if ITERATIVE_VAL:
            epoch_data['time_to_encode_corpus'] = val_metrics['time_to_encode_corpus']
            epoch_data['time_to_encode_questions'] = val_metrics['time_to_encode_questions']

            if args.ise_track_gold_indices:
                epoch_data['gold_index_eval/single_hop_average_index'] = val_metrics['gold_index_eval']['single_hop_average_index']
                epoch_data['gold_index_eval/second_hop_average_index'] = val_metrics['gold_index_eval']['second_hop_average_index']


            for (eval_name, data) in deductive_iterative_retrieval_data.items():
                data = list(
                    sorted(data, key=lambda x: x[1], reverse=True))

                #even_sampled_indices = list(set(sorted(np.round(np.linspace(0, len(deductive_residual_data) - 1, 20)).astype(int).tolist())))
                even_sampled_indices = list(range(0, len(data)))

                if WANDB:
                    iterative_search_table = wandb.Table(data=pd.DataFrame({
                        'idx': [idx for idx in even_sampled_indices],
                        'Percentage': [data[idx][0] for idx in even_sampled_indices],
                        'Target': [data[idx][4] for idx in even_sampled_indices],
                        'Args Found': [data[idx][1] for idx in even_sampled_indices],
                        'Found Arg 1': [data[idx][2][0] for idx in even_sampled_indices],
                        'Found Arg 2': [data[idx][2][1] for idx in even_sampled_indices],
                        'Gold Arg 1': [data[idx][3][0] for idx in even_sampled_indices],
                        'Gold Arg 2': [data[idx][3][1] for idx in even_sampled_indices],

                    }))

                    epoch_data[f'{eval_name}/iterative_search/iterative_search_results'] = iterative_search_table
                epoch_data[f'{eval_name}/iterative_search/all'] = val_metrics[eval_name]['all']
                epoch_data[f'{eval_name}/iterative_search/atleast_one'] = val_metrics[eval_name]['atleast_one']
                epoch_data[f'{eval_name}/iterative_search/time_for_search'] = val_metrics[eval_name]['time_to_eval']
                epoch_data[f'{eval_name}/iterative_search/time_for_enc_qs_and_search'] = val_metrics[eval_name]['time_to_eval'] + epoch_data['time_to_encode_questions']
                epoch_data[f'{eval_name}/iterative_search/total_time'] = epoch_data[f'{eval_name}/iterative_search/time_for_enc_qs_and_search'] + epoch_data['time_to_encode_corpus']

        if MRR_VAL:
            epoch_data['mrr/intermediate_mrr'] = mrr_val_metrics[0]
            epoch_data['mrr/intermediate_stddev'] = mrr_val_metrics[2]
            epoch_data['mrr/graph_goal_mrr'] = mrr_val_metrics[1]
            epoch_data['mrr/graph_goal_stddev'] = mrr_val_metrics[3]

        if WANDB:
            wandb.log(epoch_data)
        else:
            print(f'=== EPOCH {e} ===')
            print(pprint.pprint(epoch_data))

        training_batch = 0
        validation_batch = 0

        deductive_residual_data = []
        deductive_iterative_retrieval_data = {}

        epochs_since_last_best += 1

        if (ITERATIVE_VAL or MRR_VAL) and do_checkpointing:
            score = val_metrics[args.iterative_search_eval[0]]['all'] if ITERATIVE_VAL else mrr_val_metrics[1]
            if best_score is None or best_score < score:
                print("Checkpointing model...")
                best_score = score

                chkpt_path = TRAINED_MODELS_FOLDER / model_checkpoint_name
                chkpt_path.mkdir(exist_ok=True, parents=True)

                if torch.cuda.device_count() > 1 and not DEBUG:
                    model.module.save(chkpt_path / 'best_checkpoint.pth', optimizer)
                else:
                    model.save(chkpt_path / 'best_checkpoint.pth', optimizer)

                epochs_since_last_best = 0

            if epochs_since_last_best >= early_stopping_after:
                print("EARLY STOPPING")
                break
            
            ckpt_path = TRAINED_MODELS_FOLDER / model_checkpoint_name / 'perbatch' / str(e)
            ckpt_path.mkdir(exist_ok=True, parents=True)
            model.save(ckpt_path / 'best_checkpoint.pth', optimizer)


        if not Train:
            print("Stopping...")
            if WANDB:
                # Letting the stuff upload. TODO - Not sure if this is really needed
                time.sleep(60)
            break


