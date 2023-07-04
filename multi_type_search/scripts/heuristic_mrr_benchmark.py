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
from multi_type_search.search.step_selector import StepSelector
from multi_type_search.search.step_type import DeductiveStepType

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
    data: List[Graph],
    heuristic: StepSelector,
    do_intermediate_step: bool = True,
    do_graph_goal: bool = True,
    pool_premises: bool = False,
    debug: bool = False,
    tags=None,
    max_examples=-1
):
    # heuristic.model.model.eval()
    step_type = DeductiveStepType(None)

    graph_goal_ranks = []
    immediate_step_ranks = []
    max_new_steps = 0

    if pool_premises:
        premise_pool = [x for y in data for x in y.premises]
        premise_pool.extend([x for y in data for z in y.deductions for x in z.nodes])
        random.shuffle(premise_pool)
        new_steps = [x for x in step_type.generate_step_combinations(Graph(''), premise_pool) if int(x.arguments[0][8:]) < int(x.arguments[1][8:])]



    total = 0
    with torch.no_grad():
        for example in tqdm(data, desc='Running Examples', total=len(data), disable=debug):
            # if not pool_premises:
            #     premise_pool = deepcopy(example.premises)
            #     premise_pool.extend([x for y in example.deductions for x in y.nodes])
            #     random.shuffle(premise_pool)
            #     new_steps = [x for x in step_type.generate_step_combinations(Graph(''), premise_pool) if
            #              int(x.arguments[0][8:]) < int(x.arguments[1][8:])]

            for didx, deduction in enumerate(example.deductions):
                if len(deduction.arguments) != 2 or deduction.arguments[0] == deduction.arguments[1]:
                    continue
                if total >= max_examples and max_examples > -1:
                    break

                total += 1

                heuristic.reset()
                tmp_premises = deepcopy(example.premises)

                if not pool_premises:
                    premise_pool = deepcopy(example.premises)
                    premise_pool.extend([x for ddidx, y in enumerate(example.deductions) for x in y.nodes if ddidx < didx])
                    random.shuffle(premise_pool)
                    new_steps = [x for x in step_type.generate_step_combinations(Graph(''), premise_pool) if
                             int(x.arguments[0][8:]) < int(x.arguments[1][8:])]

                # allowed_premises = [x for x in premise_pool if x.normalized_value not in [example.goal.normalized_value, deduction.nodes[0].normalized_value]]
                # goal_premise = premise_pool.index(example.goal)
                # deduction_premise = premise_pool.index(deduction.nodes[0])
                # allowed_steps = [x for x in new_steps if f'PREMISE:{goal_premise}' not in list(x.arguments) and f'PREMISE:{deduction_premise}' not in list(x.arguments)]

                correct_premises = [example[x] for x in deduction.arguments]
                arg1 = premise_pool.index(correct_premises[0])
                arg2 = premise_pool.index(correct_premises[1])
                correct_step_args = {compose_index(GraphKeyTypes.PREMISE, arg1), compose_index(GraphKeyTypes.PREMISE, arg2)}

                example.premises = premise_pool

                if debug:
                    print('-'*100)

                if do_graph_goal:
                    heuristic.add_steps(new_steps, example)
                    heuristic.iter_size = len(new_steps)

                    queue = next(heuristic)
                    queue = [set(x.arguments) for x in queue]

                    if len(queue) > max_new_steps:
                        max_new_steps = len(queue)

                    try:
                        rank = queue.index(correct_step_args)
                    except Exception:
                        print('what')

                    graph_goal_ranks.append(rank)
                if do_intermediate_step:
                    heuristic.reset()

                    tmp = example.goal
                    example.goal = deduction.nodes[0]
                    heuristic.add_steps(new_steps, example)
                    heuristic.iter_size = len(new_steps)

                    raw_queue = next(heuristic)
                    queue = [set(x.arguments) for x in raw_queue]

                    if len(queue) > max_new_steps:
                        max_new_steps = len(queue)

                    rank = queue.index(correct_step_args)

                    if debug:
                        print("Deduction:")
                        print(f"\n{deduction.nodes[0].normalized_value}")
                        print("")
                        print_queue_example(example, raw_queue[rank], f'{rank+1} (C)')

                        if rank > 0:
                            for r in range(rank):
                                print_queue_example(example, raw_queue[r], r+1)
                            print_queue_example(example, raw_queue[rank], f'{rank+1} (C)')

                    immediate_step_ranks.append(rank)
                    example.goal = tmp

                example.premises = tmp_premises

    graph_goal_mrr = sum([1 / (x + 1) for x in graph_goal_ranks]) / max(1, len(graph_goal_ranks))
    immediate_step_mrr = sum([1 / (x + 1) for x in immediate_step_ranks]) / max(1, len(immediate_step_ranks))

    mrr_per_tag = {}
    if tags:
        for row, score in zip(tags, immediate_step_ranks):
            for tag in row:
                tag_ranks = mrr_per_tag.get(tag, [])
                tag_ranks.append(score)
                mrr_per_tag[tag] = tag_ranks
        for k, v in mrr_per_tag.items():
            mrr_per_tag[k] = (len(v), sum([1 / (x + 1) for x in v]) / max(1, len(v)))


    return {'intermediate_mrr': immediate_step_mrr, 'graph_goal_mrr': graph_goal_mrr}, \
           {'graph_goal_ranks': graph_goal_ranks, 'intermediate_mrr': immediate_step_ranks, 'mrr_per_tag': mrr_per_tag}


def run(
        config_path: Path,
        output_path: Path,
        max_examples: int = -1,
        force: bool = False,
        seed: int = 123,
        debug: bool = False,
        tags=None
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
    mrr_benchmark_args = merge_yaml_and_namespace(config_file, mrr_benchmark_args, ['mrr_benchmark'])

    orig_data_file = ROOT_FOLDER / mrr_benchmark_args.data_file
    if str(orig_data_file).endswith('.jsonl'):
        data = list(jsonlines.open(str(orig_data_file), 'r'))
    else:
        data = json.load(orig_data_file.open('r'))

    graphs = [Graph.from_json(t) for t in data]
    if max_examples > -1:
        rand_indices = random.sample(range(0, len(graphs)), min(max_examples, len(graphs)))
        graphs = [graphs[x] for x in rand_indices]

    with dataset_file.open('w') as file:
        json.dump([x.to_json() for x in graphs], file)

    heuristic_config = mrr_benchmark_args.heuristic
    do_intermediate_step = getattr(mrr_benchmark_args, 'do_intermediate_step', True)
    do_graph_goal = getattr(mrr_benchmark_args, 'do_graph_goal', True)
    device = getattr(mrr_benchmark_args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
    pool_premises = getattr(mrr_benchmark_args, 'pool_premises', False)

    heuristic = StepSelector.from_config('step_selector', heuristic_config, device)

    metrics, ranks = mrr(
        graphs,
        heuristic,
        do_intermediate_step=do_intermediate_step,
        do_graph_goal=do_graph_goal,
        pool_premises=pool_premises,
        debug=debug,
        tags=tags,
        max_examples=max_examples
    )

    print('-------------------------- REPORT --------------------------')
    if do_intermediate_step:
        print(f"Intermediate MRR: {metrics['intermediate_mrr']}")
    if do_graph_goal:
        print(f"Graph Goal MRR: {metrics['graph_goal_mrr']}")
    if tags:
        pprint(ranks['mrr_per_tag'])
    print('------------------------------------------------------------')

    json.dump(metrics, (output_folder / 'metrics.json').open('w'))
    json.dump(ranks, (output_folder / 'raw_ranks.json').open('w'))


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--config", '-c', required=True, type=str, help="{MTS}/configs/{path} Heuristic Benchmark Cfg.")
    parser.add_argument("--output", '-o', required=True, type=str, help="{MTS}/output/{path}")
    parser.add_argument("--max_examples", "-m", type=int, help="Maximum number of examples to run", default=-1)
    parser.add_argument("--force", '-f', action='store_true', help="Overwrite existing output.")
    parser.add_argument('--seed', '-s', type=int, default=123, help='Use this to set the random seed')
    parser.add_argument("--debug", '-d', action='store_true', help="Print stuff out")


    args = parser.parse_args()

    _config_path: Path = SEARCH_CONFIGS_FOLDER / f'{args.config}.yaml'
    _output_path: Path = SEARCH_OUTPUT_FOLDER / args.output
    _max_examples = args.max_examples
    _force: bool = args.force
    _seed: int = args.seed
    _debug: bool = args.debug

    tags = None

    run(
        config_path=_config_path,
        output_path=_output_path,
        max_examples=_max_examples,
        force=_force,
        seed=_seed,
        debug=_debug,
        tags=tags
    )


