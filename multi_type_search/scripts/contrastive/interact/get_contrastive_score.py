from argparse import ArgumentParser, Namespace
import torch
import numpy as np
import shutil
from jsonlines import jsonlines
import yaml
from typing import List, Dict
import json
from copy import deepcopy
from pathlib import Path
import random
from tqdm import tqdm

from multi_type_search.utils.paths import ROOT_FOLDER, SEARCH_OUTPUT_FOLDER, SEARCH_CONFIGS_FOLDER, TRAINED_MODELS_FOLDER
from multi_type_search.utils.config_handler import merge_yaml_and_namespace
from multi_type_search.search.graph import Graph, compose_index, GraphKeyTypes
from multi_type_search.search.step_selector import StepSelector
from multi_type_search.search.step_type import DeductiveStepType
from multi_type_search.search.search_model.types.contrastive import ContrastiveModel, NonParametricVectorSpace, \
    RobertaEncoder, RobertaMomentumEncoder, RawGPT3Encoder, ProjectedGPT3Encoder
from multi_type_search.scripts.contrastive.train_contrastive_model import create_trajectories, move_to_cuda

from argparse import ArgumentParser


def contrastive_score(model, input_strings, trajectory_creation_method: str = 'add'):
    tokens = model.tokenize(input_strings)

    if hasattr(tokens, 'data'):
        tokens.data = move_to_cuda(tokens.data)
    else:
        tokens = move_to_cuda(tokens)
    embeddings = model(tokens)
    trajectory = create_trajectories(embeddings[0], embeddings[1], trajectory_creation_method)
    score = torch.nn.functional.cosine_similarity(trajectory, embeddings[2], dim=0)
    return score


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        '--load_model_name', '-lmn', type=str,
        help='Name of the model you want to load'
    )
    parser.add_argument(
        '--trajectory_creation_method', '-tcm', type=str, default='add',
        choices=['add', 'subtract', 'multiply', 'max_pool', 'min_pool', 'avg_pool'],
        help='The method used to combine the premise embeddings ("add" is default)'
    )

    args = parser.parse_args()

    model = ContrastiveModel.load(TRAINED_MODELS_FOLDER / f'{args.load_model_name}/best_checkpoint.pth', 'cuda')
    model = model.cuda()
    while True:
        premise_1 = input('P1: ')
        if not premise_1 or premise_1 == '':
            break
        premise_2 = input('P2: ')
        if not premise_2 or premise_2 == '':
            break
        target = input('Conclusion: ')
        if not target or target == '':
            break

        input_strings = [premise_1, premise_2, target]
        score = contrastive_score(model, input_strings, args.trajectory_creation_method)
        print(f'Score: {score:.8f}')
