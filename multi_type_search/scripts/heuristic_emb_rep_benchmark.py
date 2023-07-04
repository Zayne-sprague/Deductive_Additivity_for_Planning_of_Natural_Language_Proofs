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
import matplotlib.pyplot as plt
import matplotlib
from itertools import cycle
cycol = cycle('bgrcmk')

from multi_type_search.utils.paths import ROOT_FOLDER, SEARCH_OUTPUT_FOLDER, SEARCH_CONFIGS_FOLDER
from multi_type_search.utils.config_handler import merge_yaml_and_namespace
from multi_type_search.search.graph import Graph, compose_index, GraphKeyTypes, Node
from multi_type_search.search.step_selector import StepSelector
from multi_type_search.search.step_type import DeductiveStepType, StepModel
from multi_type_search.search.search_model import NodeEmbedder


def violinplot_scores(scores: Dict[str, List[float]]):
    keys = list(scores.keys())
    vals = list(scores.values())

    def adjacent_values(vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value

    def set_axis_style(ax, labels):
        ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_xlabel('Distribution Type')
        ax.set_ylabel('Cosine Score')

    fig, (ax2) = plt.subplots(nrows=1, ncols=1, figsize=(9, 4), sharey=True)

    # fig.suptitle('Everyday Norms: Why Not?')
    fig.suptitle('Entailment Bank')

    [x.sort() for x in vals]

    parts = ax2.violinplot(
        vals, showmeans=False, showmedians=False,
        showextrema=False)

    for pc in parts['bodies']:
        # pc.set_facecolor('#0000F8')
        # pc.set_facecolor('#BF5700')
        pc.set_facecolor('#333F48')
        # pc.set_facecolor(next(cycol))
        pc.set_edgecolor('#333F48')
        pc.set_alpha(1)

    pcnts = [np.percentile(x, [25, 50, 75], axis=0) for x in vals]
    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, (q1, _, q3) in zip(vals, pcnts)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len([x[1] for x in pcnts]) + 1)
    ax2.scatter(inds, [x[1] for x in pcnts], marker='o', color='white', s=30, zorder=3)
    ax2.vlines(inds, [x[0] for x in pcnts], [x[2] for x in pcnts], color='k', linestyle='-', lw=5)
    ax2.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

    for ax in [ax2]:
        set_axis_style(ax, keys)

    plt.subplots_adjust(bottom=0.15, wspace=0.05)
    plt.show()


def rep(
    data: List[Graph],
    embedder: NodeEmbedder,
    step_model: StepModel = None,
    do_annotated: bool = True,
    do_step_model: bool = True,
    do_model_to_annotated: bool = True,
    do_random_args_to_annotated: bool = True,
    do_partial_random_args_to_annotated: bool = True
):
    data = [x for x in data if len(x.premises) > 1]
    step_type = DeductiveStepType(None)

    step_scores = []
    gold_scores = []
    stog_scores = []
    rarg_scores = []
    prarg_scores = []

    with torch.no_grad():
        for idx, example in tqdm(enumerate(data), desc='Running Examples', total=len(data)):
            for deduction in example.deductions:
                memb = None
                aemb = None
                rembs = None

                if len(deduction.arguments) != 2:
                    continue

                arg_embs = embedder.encode([example[x] for x in deduction.arguments])
                emb = arg_embs.sum(0)

                if do_step_model or do_model_to_annotated:
                    formatted_input = step_type.format_stepmodel_input([example[x].normalized_value for x in deduction.arguments])
                    step_generations = step_model.sample(formatted_input)
                    memb = embedder.encode([Node(x) for x in step_generations]).mean(0)

                if do_step_model:
                    step_scores.append(torch.nn.functional.cosine_similarity(emb, memb, 0).cpu().item())

                if do_annotated or do_model_to_annotated:
                    aemb = embedder.encode(deduction.nodes).mean(0)

                if do_random_args_to_annotated or do_partial_random_args_to_annotated:
                    rex = random.sample(list(set(range(0, len(data))) - {idx}), 1)[0]
                    rargs = random.sample(data[rex].premises, 2)
                    rembs = embedder.encode(rargs)

                if do_random_args_to_annotated:
                    rarg_scores.append(torch.nn.functional.cosine_similarity(aemb, rembs.sum(0), 0).cpu().item())

                if do_partial_random_args_to_annotated:
                    prarg_scores.append(torch.nn.functional.cosine_similarity(aemb, rembs[0, :] + arg_embs[0, :], 0).cpu().item())
                    prarg_scores.append(torch.nn.functional.cosine_similarity(aemb, rembs[1, :] + arg_embs[1, :], 0).cpu().item())
                    prarg_scores.append(torch.nn.functional.cosine_similarity(aemb, rembs[0, :] + arg_embs[1, :], 0).cpu().item())
                    prarg_scores.append(torch.nn.functional.cosine_similarity(aemb, rembs[1, :] + arg_embs[0, :], 0).cpu().item())

                if do_annotated:
                    gold_scores.append(torch.nn.functional.cosine_similarity(emb, aemb, 0).cpu().item())

                if do_model_to_annotated:
                    stog_scores.append(torch.nn.functional.cosine_similarity(aemb, memb, 0).cpu().item())

    step_avg = sum(step_scores) / max(1, len(step_scores))
    gold_avg = sum(gold_scores) / max(1, len(gold_scores))
    stog_avg = sum(stog_scores) / max(1, len(stog_scores))
    rarg_avg = sum(rarg_scores) / max(1, len(rarg_scores))
    prarg_avg = sum(prarg_scores) / max(1, len(prarg_scores))

    return {'step_avg': step_avg, 'gold_avg': gold_avg, 'stog_avg': stog_avg, 'rarg_avg': rarg_avg, 'prarg_avg': prarg_avg}, \
           {'Random': rarg_scores, 'Partially Random': prarg_scores, 'Annotation': gold_scores,  'Deduction': step_scores, 'Agreement': stog_scores}


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

    rep_benchmark_args = Namespace()
    rep_benchmark_args = merge_yaml_and_namespace(config_file, rep_benchmark_args, ['rep_benchmark'])

    orig_data_file = ROOT_FOLDER / rep_benchmark_args.data_file
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

    encoder_config = rep_benchmark_args.encoder
    step_model_config = getattr(rep_benchmark_args, 'step_model', None)
    do_annotated = getattr(rep_benchmark_args, 'do_annotated', True)
    do_step_model = getattr(rep_benchmark_args, 'do_step_model', True)
    do_model_to_annotated = getattr(rep_benchmark_args, 'do_model_to_annotated', True)
    do_random_args_to_annotated = getattr(rep_benchmark_args, 'do_random_args_to_annotated', True)
    do_partial_random_args_to_annotated = getattr(rep_benchmark_args, 'do_partial_random_args_to_annotated', True)
    show_plot = getattr(rep_benchmark_args, 'show_plot', True)
    device = getattr(rep_benchmark_args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')

    encoder = NodeEmbedder.from_config('search_model', encoder_config, device)
    step_model = StepModel.from_config('search_model', step_model_config, device)

    metrics, scores = rep(
        graphs,
        encoder, step_model,
        do_annotated=do_annotated,
        do_step_model=do_step_model,
        do_model_to_annotated=do_model_to_annotated,
        do_random_args_to_annotated=do_random_args_to_annotated,
        do_partial_random_args_to_annotated=do_partial_random_args_to_annotated
    )

    print('-------------------------- REPORT --------------------------')
    if do_annotated:
        print(f"Gold Avg Cosine Score: {metrics['gold_avg']}")
    if do_step_model:
        print(f"StepModel Avg Cosine Score: {metrics['step_avg']}")
    if do_model_to_annotated:
        print(f"StepModel and Gold Avg Cosine Score: {metrics['stog_avg']}")
    if do_random_args_to_annotated:
        print(f"Random Args to Gold Annotation Score: {metrics['rarg_avg']}")
    if do_partial_random_args_to_annotated:
        print(f"Partial Random Args to Gold Annotation Score: {metrics['prarg_avg']}")
    print('------------------------------------------------------------')

    json.dump(metrics, (output_folder / 'metrics.json').open('w'))
    json.dump(scores, (output_folder / 'scores.json').open('w'))

    if show_plot:
        violinplot_scores(scores)


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


