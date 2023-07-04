import pandas as pd
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
from pprint import pprint

from multi_type_search.utils.paths import ROOT_FOLDER, SEARCH_OUTPUT_FOLDER, SEARCH_CONFIGS_FOLDER
from multi_type_search.utils.config_handler import merge_yaml_and_namespace
from multi_type_search.search.graph import Graph, compose_index, GraphKeyTypes
from multi_type_search.search.step_selector import StepSelector, Step
from multi_type_search.search.step_type import DeductiveStepType

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def print_queue_example(
        example,
        step,
        rank
):
    print(f'Rank {rank} (Score = {step.score:.8f})')
    print(f'Value:')
    print(f"\t{example[step.arguments[0]]}")
    print(f"\t{example[step.arguments[1]]}")


def mrr(
    examples: List[Dict],
    heuristic: StepSelector,
    debug: bool = False,
):
    try:
        heuristic.model.model.activate_key()
    except Exception:
        pass
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    #heuristic.deductive_heuristic_model.model.eval()
    step_type = DeductiveStepType(None)

    ranks = {}
    totals = {}
    max_new_steps = 0

    log_out = []

    total = 0
    with torch.no_grad():
        for example in tqdm(examples, desc='Running Examples', total=len(examples), disable=debug):
            total += 1

            heuristic.reset()

            premises = list(set(example['rows']['Premise 1'].values.tolist()).union(set(example['rows']['Premise 2'].values.tolist())))
            conclusion = example['rows']['Conclusion'].values[0]
            correct_premises = [example['rows']['Premise 1'].values[0], example['rows']['Premise 2'].values[0]]
            random.shuffle(premises)

            arg1 = premises.index(correct_premises[0])
            arg2 = premises.index(correct_premises[1])

            correct_step_args = {compose_index(GraphKeyTypes.PREMISE, arg1), compose_index(GraphKeyTypes.PREMISE, arg2)}

            graph = Graph(conclusion, premises)

            new_steps = []
            for (_, row) in [x for x in example['rows'].iterrows()]:
                if row['Conclusion'] != conclusion:
                    continue
                p1 = premises.index(row['Premise 1'])
                p2 = premises.index(row['Premise 2'])

                step_args = {compose_index(GraphKeyTypes.PREMISE, p1), compose_index(GraphKeyTypes.PREMISE, p2)}
                if len(step_args) != 2:
                    continue

                new_steps.append(Step(step_args, step_type))

            if len(new_steps) < 2:
                continue

            random.shuffle(new_steps)

            heuristic.add_steps(new_steps, graph)
            heuristic.iter_size = len(new_steps)

            queue = next(heuristic)
            queue = [set(x.arguments) for x in queue]

            if len(queue) > max_new_steps:
                max_new_steps = len(queue)

            rank = queue.index(correct_step_args)

            reasoning_category = example['rows']['Reason Type'].values[0]
            perturbation_type = example['category']

            group = ranks.get(reasoning_category, {})
            perturb_ranks = group.get(perturbation_type, [])
            perturb_ranks.append(rank)
            group[perturbation_type] = perturb_ranks
            ranks[reasoning_category] = group

            group = totals.get(reasoning_category, {})
            perturb_ranks = group.get(perturbation_type, [])
            perturb_ranks.append(len(new_steps))
            group[perturbation_type] = perturb_ranks
            totals[reasoning_category] = group

            log_out.append(f'EX {total}\n')
            log_out.append(f'Category: {reasoning_category}\n')
            log_out.append(f'Perturbation Type: {perturbation_type}\n')
            log_out.append(f'Target: {conclusion}\n')
            log_out.append(f'Gold Premises: {" ".join(correct_premises)}\n')
            log_out.append(f'Rank: {rank+1}\n')
            for idx, s in enumerate(queue):
                ex_str = " ".join([graph[x].normalized_value for x in s])
                if idx == rank:
                    log_out.append(f"\tRank (G) {idx + 1}: {ex_str}\n")
                else:
                    log_out.append(f"\tRank {idx + 1}: {ex_str}\n")
            log_out.append('\n\n')

    metrics = {}
    raw_ranks = {}
    for reasoning_category, perturbation_groups in ranks.items():
        metrics[reasoning_category] = {}
        raw_ranks[reasoning_category] = {}
        for perturbation_type, perturbation_ranks in perturbation_groups.items():
            metrics[reasoning_category][perturbation_type] = [
                sum([1 / (x + 1) for x in perturbation_ranks]) / max(1, len(perturbation_ranks)),
                len(perturbation_ranks)
            ]
            raw_ranks[reasoning_category][perturbation_type] = [f'{str(x+1)}/{y}' for x, y in zip(perturbation_ranks, totals[reasoning_category][perturbation_type])]

    print("".join(log_out))
    with open('logout.txt', 'w') as f:
        f.write("".join(log_out))

    return metrics, raw_ranks


def run(
        config_path: Path,
        output_path: Path,
        force: bool = False,
        seed: int = 123,
        debug: bool = False,
):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    data_folder = output_path / 'data'
    # vis_folder = output_path / 'visualizations'
    # reports_folder = output_path / 'reports'
    output_folder = output_path / 'output'

    assert not output_folder.exists() or force, \
        'Please specify an empty folder path for the output parameter -o OR specify the force flag -f'

    if output_folder and output_path.exists():
        shutil.rmtree(str(output_path))

    data_folder.mkdir(exist_ok=False, parents=True)
    # vis_folder.mkdir(exist_ok=False, parents=True)
    # reports_folder.mkdir(exist_ok=False, parents=True)
    output_folder.mkdir(exist_ok=False, parents=True)

    config_file = output_path / 'config.yaml'
    dataset_file = data_folder / 'experiment_dataset.json'

    shutil.copyfile(str(config_path), str(config_file))

    mrr_benchmark_args = Namespace()
    mrr_benchmark_args = merge_yaml_and_namespace(config_file, mrr_benchmark_args, ['mrr_reasoning_benchmark'])

    orig_data_file = ROOT_FOLDER / mrr_benchmark_args.data_file

    # TODO: GPT4 Read ORIG DATA FILE CSV
    data = pd.read_csv(orig_data_file)

    # Filter only rows with the "Label" column equal to "T" or "F"
    data = data[data['Label'].isin(['T', 'F'])]
    perturbation_values = ["NEGATED", "IRRELEVANT FACT", "FALSE PREMISE", "Incorrect Quantifier"]

    # Create a column 'Perturbation_category' to store the category of each perturbation
    data['Perturbation_category'] = ''
    for perturbation in perturbation_values:
        data.loc[data['Perturbation'].str.contains(perturbation), 'Perturbation_category'] = perturbation

    # Find the indices of rows with the "Label" column equal to "T"
    t_indices = data[data['Label'] == 'T'].index

    # Create an empty list to store the output
    output = []

    # Iterate through the "T" indices and group the rows as required
    for i, t_index in enumerate(t_indices):
        start_index = t_index
        end_index = t_indices[i + 1] if i + 1 < len(t_indices) else len(data)

        # Extract the rows between the current and the next "T" index
        example_data = data.iloc[start_index:end_index]

        # Group the rows by the 'Perturbation_category' column and add the "T" row to each group
        grouped_data = example_data.groupby('Perturbation_category')
        for group_name, group_data in grouped_data:
            if group_name != '':
                t_row = data.iloc[start_index]
                group_rows = group_data.reset_index(drop=True)
                output.append(
                    {'category': group_name, 'rows': pd.concat([t_row.to_frame().T, group_rows], ignore_index=True)})

    # Now, the 'output' list contains the grouped data as required

    heuristic_config = mrr_benchmark_args.heuristic
    device = getattr(mrr_benchmark_args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
    heuristic = StepSelector.from_config('step_selector', heuristic_config, device)

    metrics, ranks = mrr(
        output,
        heuristic,
        debug=debug,
    )

    print('-------------------------- REPORT --------------------------')
    pprint(ranks)
    pprint(metrics)
    print('------------------------------------------------------------')




if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--config", '-c', required=True, type=str, help="{MTS}/configs/{path} Heuristic Benchmark Cfg.")
    parser.add_argument("--output", '-o', required=True, type=str, help="{MTS}/output/{path}")
    parser.add_argument("--force", '-f', action='store_true', help="Overwrite existing output.")
    parser.add_argument('--seed', '-s', type=int, default=123, help='Use this to set the random seed')
    parser.add_argument("--debug", '-d', action='store_true', help="Print stuff out")


    args = parser.parse_args()

    _config_path: Path = SEARCH_CONFIGS_FOLDER / f'{args.config}.yaml'
    _output_path: Path = SEARCH_OUTPUT_FOLDER / args.output
    _force: bool = args.force
    _seed: int = args.seed
    _debug: bool = args.debug

    run(
        config_path=_config_path,
        output_path=_output_path,
        force=_force,
        seed=_seed,
        debug=_debug,
    )


