import numpy as np
import time
import pandas as pd
import torch
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader
from typing import Iterable, Callable, List, Union
from copy import deepcopy
import apex

apex.amp.register_half_function(torch, 'einsum')
from apex import amp
from pathlib import Path
import shutil

from multi_type_search.search.search_model.types.contrastive import ContrastiveModel, NonParametricVectorSpace, \
    RobertaEncoder, RobertaMomentumEncoder
from multi_type_search.search.search_model.types.contrastive.contrastive_utils import cosine_similarity_metric
from multi_type_search.scripts.contrastive.contrastive_losses import ContrastiveLoss
from multi_type_search.scripts.contrastive.contrastive_dataset import ContrastiveDataset
from multi_type_search.scripts.contrastive.corpus_dataset import CorpusDataset
from multi_type_search.scripts.contrastive.goals_dataset import GoalDataset
from multi_type_search.scripts.contrastive.evaluate import iterative_search, rank_support_sets, \
    multiple_support_set_metrics, iterative_search__both, faiss_iterative_search, iterative_search__subt


def inbatch_comparision_analysis(trajectories, deductive_goal_embs, arg1, arg2, string_goal, string_args):

    mse_table = ((trajectories - deductive_goal_embs.unsqueeze(1)) ** 2).sum(-1).cpu().numpy()
    mse_index_table = mse_table.argsort(-1)
    cosine_table = cosine_similarity_metric(trajectories, deductive_goal_embs.unsqueeze(1)).cpu().numpy()
    cosine_index_table = cosine_table.argsort(-1)[:, ::-1]

    print("HI")


def handle_batch(
        model,
        batch
):
    _, _, args, deductive_goal, _, string_args, string_goal, _ = batch

    args = to_cuda(args)
    deductive_goal = to_cuda(deductive_goal)

    arg_embs, _, _ = model(args)
    deductive_goal_embs, _, _ = model(deductive_goal)

    arg_embs = arg_embs.view([deductive_goal_embs.shape[0], 2, graph_goal_embs.shape[-1]])
    arg1 = arg_embs[:, 0, :]
    arg2 = arg_embs[:, 1, :]
    trajectories = arg1 + arg2

    return trajectories, deductive_goal_embs, arg1, arg2, string_goal, string_args


def handle_batch_with_momentum(
        model: RobertaMomentumEncoder,
        batch,
        use_memory_bank: bool = False
):

    _, _, args, deductive_goal, _, string_args, string_goal, _ = batch

    args = to_cuda(args)
    deductive_goal = to_cuda(deductive_goal)

    if use_memory_bank:
        arg_embs = model.q_enc(args)
    else:
        arg_embs = model.k_enc(args)

    deductive_goal_embs = model.q_enc(deductive_goal)

    arg_embs = arg_embs.view([-1, 2, deductive_goal_embs.shape[-1]])
    arg1 = arg_embs[:, 0, :]
    arg2 = arg_embs[:, 1, :]

    trajectories = arg1 + arg2

    return trajectories, deductive_goal_embs, arg1, arg2, string_goal, string_args


def to_cuda(x):
    """Helper """
    if torch.cuda.is_available():
        if isinstance(x, object) and not isinstance(x, dict):
            x.data['input_ids'] = x.data['input_ids'].cuda()
            x.data['attention_mask'] = x.data['attention_mask'].cuda()
            return x
        if isinstance(x, dict):
            return {k: v.cuda() for k, v in x.items()}
        return x.cuda()
    return x


def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda()
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


@torch.no_grad()
def analyze_examples(
        model: Union[NonParametricVectorSpace, RobertaMomentumEncoder, RobertaEncoder],
        dataloader: DataLoader,
        use_momentum: bool = False,
        use_memory_bank: bool = False,

):
    model.eval()

    total_batches = len(dataloader)

    for idx, batch in tqdm(enumerate(dataloader), total=total_batches, desc='Analyzing vector space examples...'):
        if use_momentum or use_memory_bank:
            trajectories, deductive_goal_embs, arg1, arg2, string_goal, string_args = handle_batch_with_momentum(model, batch,
                                                                                                        use_memory_bank)
        else:
            trajectories, deductive_goal_embs, arg1, arg2, string_goal, string_args = handle_batch(model, batch)

        inbatch_comparision_analysis(trajectories, deductive_goal_embs, arg1, arg2, string_goal, string_args)


if __name__ == "__main__":
    from multi_type_search.scripts.contrastive.contrastive_losses import NTXENTLoss, CosineLoss, MOCOLoss
    from multi_type_search.utils.paths import DATA_FOLDER, TRAINED_MODELS_FOLDER
    from multi_type_search.search.graph import Graph

    from torch import optim
    from torch.nn import DataParallel

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
        '--data_file', '-df', type=str, default=str(DATA_FOLDER / 'full/hotpot/fullwiki_train.json'),
        help='Path to the training file'
    )
    parser.add_argument('--batch_size', '-bs', type=int, help='Size of batch in training', default=50)


    parser.add_argument(
        '--max_data_size', '-mds', type=int, default=None,
        help='Maximum number of training graphs to use.  None means use all.'
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
        '--momentum_queue_size', '-mqs', type=int, default=2048,
        help='How large the momentum queue should be'
    )

    args = parser.parse_args()

    # DEBUG: Controls if pin memory / num workers in the dataloader
    DEBUG = args.debug


    # Local paths from the top level data directory to training and validation files
    data_file = Path(args.data_file)

    # Maximum sizes of the number of Graphs loaded per dataset.  None means all of the graphs.
    MAX_DATA_SIZE = args.max_data_size

    if args.use_momentum_encoder or args.use_memory_bank:
        roberta_model = RobertaEncoder()
        model = RobertaMomentumEncoder(
            roberta_model,
            args.momentum_queue_size,
            not args.use_memory_bank
        )
    else:
        model = NonParametricVectorSpace(
            roberta_base=True,
            t5_token_max_length=256,
            fw_hidden_size=100,
            embedding_size=100,
            residual_connections=False,
            backbone_only=True,
            freeze_backbone=False
        )

    if args.load_model_name:
        model = model.load(TRAINED_MODELS_FOLDER / f'{args.load_model_name}/best_checkpoint.pth', 'cuda')

    graphs = [Graph.from_json(x) for x in (json.load(data_file.open('r')) if data_file.name.endswith(
        '.json') else list(jsonlines.Reader(data_file.open('r'))))[:MAX_DATA_SIZE]]

    dataset = ContrastiveDataset(tokenizer=model.roberta_tokenizer)

    dataset.populate_examples(graphs)

    if torch.cuda.is_available():
        model = model.cuda()

    dataloader = DataLoader(
        dataset, args.batch_size, shuffle=False, collate_fn=dataset.collate_fn,
        pin_memory=not DEBUG, num_workers=1 if DEBUG else 20
    )
    # ACTUAL TRAINING LOOP

    analyze_examples(
        model=model,
        dataloader=dataloader,
        use_momentum=args.use_momentum_encoder,
        use_memory_bank=args.use_memory_bank,
    )
