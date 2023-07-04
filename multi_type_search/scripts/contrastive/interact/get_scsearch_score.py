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
from multi_type_search.search.step_selector import StepSelector, Step
from multi_type_search.search.step_type import DeductiveStepType
from multi_type_search.search.search_model import CalibratorHeuristic

from multi_type_search.search.search_model.types.contrastive import ContrastiveModel, NonParametricVectorSpace, \
    RobertaEncoder, RobertaMomentumEncoder, RawGPT3Encoder, ProjectedGPT3Encoder
from multi_type_search.scripts.contrastive.train_contrastive_model import create_trajectories, move_to_cuda

from argparse import ArgumentParser



def scsearch_score(model, input_strings, *args, **kwargs):
    assert len(input_strings) == 3, 'Incorrect number of premises.'

    g = Graph(goal=input_strings[2], premises=input_strings[0:2])
    step = Step(arguments=['PREMISE:0', 'PREMISE:1'], type=None)
    score = model.score_steps(g, [step])[0]
    return score


if __name__ == "__main__":

    model = CalibratorHeuristic('forward_v3_gc', device='cuda', goal_conditioned=True)

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

        score = scsearch_score(model, [premise_1, premise_2, target])

        print(f'Score: {score:.8f}')
