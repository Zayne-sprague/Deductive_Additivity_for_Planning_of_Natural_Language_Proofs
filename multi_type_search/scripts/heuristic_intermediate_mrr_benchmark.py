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
import time
import matplotlib.pyplot as plt
import matplotlib

from multi_type_search.utils.paths import ROOT_FOLDER, SEARCH_OUTPUT_FOLDER, SEARCH_CONFIGS_FOLDER
from multi_type_search.utils.config_handler import merge_yaml_and_namespace
from multi_type_search.search.graph import Graph, Node, HyperNode, compose_index, GraphKeyTypes, HyperNodeTypes
from multi_type_search.search.step_selector import StepSelector
from multi_type_search.search.step_type import DeductiveStepType, StepType
from multi_type_search.search.generation_validator import ContrastiveFilterValidator


def visualize_ranks(
        metrics,
        do_graph_goal: bool = False,
        do_intermediate_step: bool = False
):
    x_axis = list(range(len(metrics['graph_goal_mrr']))) if do_graph_goal else list(range(len(metrics['intermediate_mrr'])))

    if do_graph_goal:
        plt.plot(x_axis, metrics['graph_goal_mrr'], '-.', label='graph goal')
    if do_intermediate_step:
        plt.plot(x_axis, metrics['intermediate_mrr'], label='intermediate')

    plt.xlabel("Added Generations")
    plt.ylabel("MRR")
    plt.legend()
    plt.title('MRR vs # Of Generations')
    plt.show()


def get_rank(
    heuristic,
    new_steps,
    example,
    correct_step_args
):
    n = time.time()
    heuristic.add_steps(new_steps, example)
    heuristic.iter_size = len(new_steps)

    queue = next(heuristic)
    queue = [set(x.arguments) for x in queue]

    rank = queue.index(correct_step_args)
    return rank, time.time() - n


def intermediate_mrr(
    data: List[Graph],
    heuristic: StepSelector,
    step_type: StepType,
    do_intermediate_step: bool = True,
    do_graph_goal: bool = True,
    random_step_samples: int = 10,
    contrastive_filter: ContrastiveFilterValidator = None,
    ranks_per_added_generation: bool = False,
):
    added_nodes = []
    graph_goal_ranks = []
    graph_goal_times = []
    immediate_step_ranks = []
    immediate_times = []

    with torch.no_grad():
        for example in tqdm(data, desc='Running Examples', total=len(data)):
            deduction = example.deductions[0]

            # Early exit if the annotated deduction is weirdly formatted.
            if len(deduction.arguments) != 2 or deduction.arguments[0] == deduction.arguments[1]:
                continue

            orig_premises = deepcopy(example.premises)
            premise_pool = deepcopy(example.premises)

            # Do not randomly sample from the gold step (doesn't make sense to do MRR when we have sampled from it)
            non_gold_set = [x for x in premise_pool if x.normalized_value != example[deduction.arguments[0]].normalized_value and x.normalized_value != example[deduction.arguments[1]].normalized_value]
            correct_premises = [example[x] for x in deduction.arguments]

            if contrastive_filter:
                example.premises = non_gold_set

            # Create generations from a random selection of premises.
            for i in range(random_step_samples):
                bad_arg = random.sample(non_gold_set, 1)
                correct_arg = random.sample(correct_premises, 1)
                step = [bad_arg[0], correct_arg[0]]
                generations = step_type.step_model.sample(step_type.format_stepmodel_input([x.normalized_value for x in step]))
                new_nodes = [Node(x) for x in generations]
                new_nodes = [x for x in new_nodes if x.normalized_value not in [correct_premises[0].normalized_value, correct_premises[1].normalized_value, deduction.nodes[0].normalized_value]]
                premise_pool.extend(new_nodes)
                added_nodes.extend(new_nodes)

                if contrastive_filter:
                    ded = HyperNode(HyperNodeTypes.Deductive, new_nodes, [compose_index(GraphKeyTypes.PREMISE, non_gold_set.index(x)) for x in step])
                    contrastive_filter.validate(example, None, [], [], [ded])

            if contrastive_filter:
                example.premises = orig_premises

            # Shuffle the premises and create steps that we can rank using the heuristic.
            random.shuffle(premise_pool)
            new_steps = [x for x in step_type.generate_step_combinations(Graph(''), premise_pool) if
                     int(x.arguments[0][8:]) < int(x.arguments[1][8:])]

            # Ensure the heuristic is not contaminated with the previous iterations' data.
            heuristic.reset()

            # Construct the correct step we should rank highly using the heuristic.
            arg1 = premise_pool.index(correct_premises[0])
            arg2 = premise_pool.index(correct_premises[1])
            correct_step_args = {compose_index(GraphKeyTypes.PREMISE, arg1), compose_index(GraphKeyTypes.PREMISE, arg2)}

            # Reset the example premises pool with the new generations we randomly sampled from before.
            # WARNING: after doing this, the deduction.arguments for each deduction will be wrong.
            example.premises = premise_pool

            # Either get the MRR with all the added premises in one go, or, get the MRR as we add the generations one
            # by one.
            if ranks_per_added_generation:
                gg_ranks = []
                gg_times = []
                int_ranks = []
                int_times = []

                random.shuffle(added_nodes)

                for i in range(len(added_nodes), -1, -1):

                    allowed_steps = [x for x in new_steps if example[x.arguments[0]] not in added_nodes[:i] and example[x.arguments[1]] not in added_nodes[:i]]

                    if do_graph_goal:
                        heuristic.reset()

                        rank, t = get_rank(heuristic, allowed_steps, example, correct_step_args)

                        gg_ranks.append(rank)
                        gg_times.append(t)
                    if do_intermediate_step:
                        heuristic.reset()

                        tmp = example.goal
                        example.goal = deduction.nodes[0]

                        rank, t = get_rank(heuristic, allowed_steps, example, correct_step_args)

                        int_ranks.append(rank)
                        int_times.append(t)
                        example.goal = tmp

                graph_goal_ranks.append(gg_ranks)
                graph_goal_times.append(gg_times)
                immediate_step_ranks.append(int_ranks)
                immediate_times.append(int_times)

            else:
                if do_graph_goal:
                    heuristic.reset()

                    rank, t = get_rank(heuristic, new_steps, example, correct_step_args)

                    graph_goal_ranks.append(rank)
                    graph_goal_times.append(t)
                if do_intermediate_step:
                    heuristic.reset()

                    tmp = example.goal
                    example.goal = deduction.nodes[0]

                    rank, t = get_rank(heuristic, new_steps, example, correct_step_args)

                    immediate_step_ranks.append(rank)
                    immediate_times.append(t)
                    example.goal = tmp

    if ranks_per_added_generation:
        graph_goal_mrr = []
        graph_goal_time = []
        immediate_step_mrr = []
        immediate_time = []

        for idx in range(len(graph_goal_ranks[0])):
            gg_ranks = [x[idx] for x in graph_goal_ranks]
            gg_times = [x[idx] for x in graph_goal_times]
            int_ranks = [x[idx] for x in immediate_step_ranks]
            int_times = [x[idx] for x in immediate_times]

            graph_goal_mrr.append(sum([1 / (x + 1) for x in gg_ranks]) / max(1, len(gg_ranks)))
            graph_goal_time.append(sum(gg_times) / max(1, len(gg_times)))
            immediate_step_mrr.append(sum([1 / (x + 1) for x in int_ranks]) / max(1, len(int_ranks)))
            immediate_time.append(sum(int_times) / max(1, len(int_times)))
    else:
        graph_goal_mrr = sum([1 / (x + 1) for x in graph_goal_ranks]) / max(1, len(graph_goal_ranks))
        graph_goal_time = sum(graph_goal_times) / max(1, len(graph_goal_times))
        immediate_step_mrr = sum([1 / (x + 1) for x in immediate_step_ranks]) / max(1, len(immediate_step_ranks))
        immediate_time = sum(immediate_times) / max(1, len(immediate_times))

    return {'intermediate_mrr': immediate_step_mrr, 'graph_goal_mrr': graph_goal_mrr, 'intermediate_time': immediate_time, 'graph_goal_time': graph_goal_time}, \
           {'graph_goal_ranks': graph_goal_ranks, 'intermediate_mrr': immediate_step_ranks, 'graph_goal_times': graph_goal_times, 'intermediate_times': immediate_times}


def run(
        config_path: Path,
        output_path: Path,
        max_examples: int = -1,
        force: bool = False,
        seed: int = 123
):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    data_folder = output_path / 'data'
    output_folder = output_path / 'output'

    assert not output_folder.exists() or force, \
        'Please specify an empty folder path for the output parameter -o OR specify the force flag -f'

    if output_folder and output_path.exists():
        shutil.rmtree(str(output_path))

    data_folder.mkdir(exist_ok=False, parents=True)
    output_folder.mkdir(exist_ok=False, parents=True)

    config_file = output_path / 'config.yaml'
    dataset_file = data_folder / 'experiment_dataset.json'

    shutil.copyfile(str(config_path), str(config_file))

    mrr_benchmark_args = Namespace()
    mrr_benchmark_args = merge_yaml_and_namespace(config_file, mrr_benchmark_args, ['mrr_intermediate_benchmark'])

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
    step_type_config = mrr_benchmark_args.step_type
    contrastive_filter_config = getattr(mrr_benchmark_args, 'contrastive_validator', None)
    do_intermediate_step = getattr(mrr_benchmark_args, 'do_intermediate_step', True)
    do_graph_goal = getattr(mrr_benchmark_args, 'do_graph_goal', True)
    random_step_samples = getattr(mrr_benchmark_args, 'random_step_samples', 10)
    ranks_per_added_generation = getattr(mrr_benchmark_args, 'ranks_per_added_generation', True)
    visualize = getattr(mrr_benchmark_args, 'visualize', True)
    device = getattr(mrr_benchmark_args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')

    heuristic = StepSelector.from_config('step_selector', heuristic_config, device)
    step_type = StepType.from_config('step_type', step_type_config, device)
    contrastive_filter = ContrastiveFilterValidator.from_config('generation_validator', contrastive_filter_config, device) if contrastive_filter_config else None

    metrics, ranks = intermediate_mrr(
        graphs,
        heuristic,
        step_type,
        do_intermediate_step=do_intermediate_step,
        do_graph_goal=do_graph_goal,
        random_step_samples=random_step_samples,
        contrastive_filter=contrastive_filter,
        ranks_per_added_generation=ranks_per_added_generation
    )

    print('-------------------------- REPORT --------------------------')
    if ranks_per_added_generation:
        for i in range(len(metrics['intermediate_mrr'])):
            if do_intermediate_step:
                print(f'Intermediate MRR @ {i}: {metrics["intermediate_mrr"][i]}')
                print(f'Intermediate MRR @ {i}: {metrics["intermediate_time"][i]}')
            if do_graph_goal:
                print(f"Graph Goal MRR @ {i}: {metrics['graph_goal_mrr'][i]}")
                print(f"Graph Goal Time @ {i}: {metrics['graph_goal_time'][i]}")
    else:
        if do_intermediate_step:
            print(f"Intermediate MRR: {metrics['intermediate_mrr']}")
            print(f"Intermediate Time: {metrics['intermediate_time']}")
        if do_graph_goal:
            print(f"Graph Goal MRR: {metrics['graph_goal_mrr']}")
            print(f"Graph Goal Time: {metrics['graph_goal_time']}")
    print('------------------------------------------------------------')

    json.dump(metrics, (output_folder / 'metrics.json').open('w'))
    json.dump(ranks, (output_folder / 'raw_ranks.json').open('w'))

    if visualize and ranks_per_added_generation:
        visualize_ranks(metrics, do_graph_goal, do_intermediate_step)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--config", '-c', required=True, type=str, help="{MTS}/configs/{path} Heuristic Benchmark Cfg.")
    parser.add_argument("--output", '-o', required=True, type=str, help="{MTS}/output/{path}")
    parser.add_argument("--max_examples", "-m", type=int, help="Maximum number of examples to run", default=-1)
    parser.add_argument("--force", '-f', action='store_true', help="Overwrite existing output.")
    parser.add_argument('--seed', '-s', type=int, default=123, help='Use this to set the random seed')

    args = parser.parse_args()

    _config_path: Path = SEARCH_CONFIGS_FOLDER / f'{args.config}.yaml'
    _output_path: Path = SEARCH_OUTPUT_FOLDER / args.output
    _max_examples = args.max_examples
    _force: bool = args.force
    _seed: int = args.seed

    run(
        config_path=_config_path,
        output_path=_output_path,
        max_examples=_max_examples,
        force=_force,
        seed=_seed
    )


