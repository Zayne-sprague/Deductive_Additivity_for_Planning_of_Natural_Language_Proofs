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
    RobertaEncoder, RobertaMomentumEncoder, RawGPT3Encoder, ProjectedGPT3Encoder, XCSE
from multi_type_search.scripts.contrastive.train_contrastive_model import create_trajectories, move_to_cuda
from multi_type_search.scripts.contrastive.interact.get_contrastive_score import contrastive_score
from multi_type_search.scripts.contrastive.interact.get_gpt3_score import gpt3_score
from multi_type_search.scripts.contrastive.interact.get_scsearch_score import scsearch_score
from multi_type_search.search.search_model import CalibratorHeuristic

from argparse import ArgumentParser

model_names = []
models = []
model_score_fns = []

outputs = []


def parse_gpt3_false_premise_template(response: str):

    lines = response.split('\n')
    lines = [x for x in lines if x != '']

    block_size = 6
    for i in range(0,len(lines),block_size):
        block = lines[i:i+block_size]
        p1 = block[1].replace('Premise 1:', '')
        p2 = block[2].replace('Premise 2:', '')
        ded = block[3].replace('Conclusion:', '')
        p1n = block[4].replace('Premise 3 (FALSE PREMISE):', '')
        p2n = block[5].replace('Premise 4 (FALSE PREMISE):', '')

        interaction_loop(p1, p2, ded, 'T', 'Syllogism')
        interaction_loop(p1n, p2, ded, 'F', 'Syllogism', 'False Premise')
        interaction_loop(p1, p2n, ded, 'F', 'Syllogism', 'False Premise')
        interaction_loop(p1n, p2n, ded, 'F', 'Syllogism', 'False Premise, False Premise')

    print('\n' * 10)
    for output in outputs:
        line = ";".join(output)
        print(f'{line}')


def parse_gpt3_negation_template(response: str):

    lines = response.split('\n')
    lines = [x for x in lines if x != '']

    block_size = 7
    for i in range(0,len(lines),block_size):
        block = lines[i:i+block_size]
        p1 = block[1].replace('Premise 1:', '')
        p2 = block[2].replace('Premise 2:', '')
        ded = block[3].replace('Conclusion 1:', '')
        p1n = block[4].replace('Premise 3 (NEGATED):', '')
        p2n = block[5].replace('Premise 4 (NEGATED):', '')
        dedn = block[6].replace('Conclusion 2 (NEGATED):', '')

        interaction_loop(p1, p2, ded, 'T', 'Syllogism')
        interaction_loop(p1n, p2, ded, 'F', 'Syllogism', 'Negated')
        interaction_loop(p1, p2n, ded, 'F', 'Syllogism', 'Negated')
        interaction_loop(p1n, p2n, ded, 'F', 'Syllogism', 'Negated, Negated')
        interaction_loop(p1, p2, dedn, 'F', 'Syllogism', 'Negated (C)')

    print('\n' * 10)
    for output in outputs:
        line = ";".join(output)
        print(f'{line}')

def parse_gpt3_3bad_conc_template(response: str):

    lines = response.split('\n')
    lines = [x for x in lines if x != '']

    block_size = 7
    for i in range(0,len(lines),block_size):
        block = lines[i:i+block_size]
        p1 = block[1].replace('P1:', '')
        p2 = block[2].replace('P2:', '')
        ded = block[3].replace('C1:', '')
        dedum = block[4].replace('C2:', '').replace('(UNDISTRIBUTED MIDDLE)', '')
        dedhg = block[5].replace('C3:', '').replace('(HASTY GENERALIZATION)', '')
        dedus = block[6].replace('C4:', '').replace('(UNREPRESENTATIVE SAMPLE)', '')

        interaction_loop(p1, p2, ded, 'T', 'Syllogism')
        interaction_loop(p1, p2, dedum, 'F', 'Syllogism', 'Undistributed Middle')
        interaction_loop(p1, p2, dedhg, 'F', 'Syllogism', 'Hasty Generalization')
        interaction_loop(p1, p2, dedus, 'F', 'Syllogism', 'Unrepresentative Sample')

    print('\n' * 10)
    for output in outputs:
        line = ";".join(output)
        print(f'{line}')


def interaction_loop(
        p1: str = None,
        p2: str = None,
        ded: str = None,
        label: str = None,
        ded_type: str = None,
        error_label: str = None
):
    if not p1 or not p2 or not ded or not label:

        p1 = input("P1: ")
        if p1 == 'print':
            print('\n'*10)
            for output in outputs:
                line = ";".join(output)
                print(f'{line}')
                return
        p2 = input("P2: ")
        ded = input("Conclusion: ")
        while True:
            label = input("label (T or F): ")
            if label == 'T' or label == 'F':
                break

    p1 = p1.lower().strip()
    p2 = p2.lower().strip()
    ded = ded.lower().strip()

    output = [p1, p2, ded, label]

    print(f"{p1} + {p2} -> {ded}")
    for name, model, score_fn in zip(model_names, models, model_score_fns):
        score = score_fn(model, [p1, p2, ded], "add")
        print(f'\t{name}: {score:.8f}')
        output.append(f'{score:.8f}')
    if ded_type:
        output.append(ded_type)
    if error_label:
        output.append(error_label)
    print("")
    outputs.append(output)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        '--load_model_names', '-lmn', type=str, nargs='+',
        help='Name of the model you want to load.  Put "{GPT3|SCSEARCH|CONTRASTIVE}:" infront of the model name so we know'
             ' what scoring function to use.'
    )

    parser.add_argument(
        '--text_files', '-tf', type=str, nargs='+',
        help='Name of file with text to parse, use the format {FORMAT:FILENAME} where {FORMAT} can be {GPT_NEGATION}.'
    )

    args = parser.parse_args()

    loading_model_names = args.load_model_names
    text_files = args.text_files

    for name in loading_model_names:
        _type, _name = name.split(':')
        model_names.append(_name)

        if _type == 'GPT3':
            if _name.lower() == 'raw':
                models.append(RawGPT3Encoder())
            else:
                models.append(ContrastiveModel.load(TRAINED_MODELS_FOLDER / f'{_name}/best_checkpoint.pth',
                                              'cuda').cuda())
            models[-1].activate_key()
            model_score_fns.append(gpt3_score)
        elif _type == "SCSEARCH":
            models.append(CalibratorHeuristic(_name, device='cuda', goal_conditioned=True))
            model_score_fns.append(scsearch_score)
        elif _type == "CONTRASTIVE":
            if _name == 'simcse':
                models.append(XCSE('SimCSE', 'princeton-nlp/unsup-simcse-roberta-large'))
            else:
                models.append(ContrastiveModel.load(TRAINED_MODELS_FOLDER / f'{_name}/best_checkpoint.pth',
                                                    'cuda').cuda())
            model_score_fns.append(contrastive_score)

    if text_files:
        for f in text_files:
            template, name = f.split(":")
            txt = Path(name).open('r').read()

            if template == 'GPT_NEGATION':
                parse_gpt3_negation_template(txt)
            elif template == "GPT_FALSE_PREMISE":
                parse_gpt3_false_premise_template(txt)
            elif template == "GPT_BAD_CONCS":
                parse_gpt3_3bad_conc_template(txt)
            else:
                print(f"ERROR COULDN'T PARSE: {f}")

    while True:
        interaction_loop()
