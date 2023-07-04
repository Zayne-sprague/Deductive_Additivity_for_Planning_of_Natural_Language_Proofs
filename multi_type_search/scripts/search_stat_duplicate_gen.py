from multi_type_search.search.graph import Node, HyperNode, HyperNodeTypes, Graph, GraphKeyTypes, decompose_index, \
    compose_index
from multi_type_search.search.comparison_metric import ExactComparison, RougeEntailmentHMComparison, RougeComparison, \
    EntailmentComparison, EntailmentModel, EntailmentMethod, ComparisonMetric
from multi_type_search.utils.paths import SEARCH_OUTPUT_FOLDER

import os
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import json
from jsonlines import jsonlines
from argparse import ArgumentParser
from copy import deepcopy
from tqdm import tqdm
import statistics
import matplotlib.pyplot as plt
import csv


def search_stat_duplicate_generations(
        experiment_path: Path,
        device: str = 'cpu',
        track_examples: bool = False
):
    search_file = experiment_path / 'output/searched.json'
    graphs = [Graph.from_json(x) for x in json.load(search_file.open('r'))]

    duplicate_counts = {}
    percentage_of_duplicates = 0.

    de_exact_step_arg_dupes = []
    ab_exact_step_arg_dupes = []
    de_exact_step_node_dupes = []
    ab_exact_step_node_dups = []

    de_rehm_step_arg_dupes = []
    ab_rehm_step_arg_dupes = []
    de_rehm_step_node_dupes = []
    ab_rehm_step_node_dupes = []

    rouge_metric = RougeComparison(rouge_types=['rouge1'])
    entailment_model = EntailmentModel('wanli_entailment_model', device=device)
    entailment_metric = EntailmentComparison(entailment_model, EntailmentMethod.mutual)
    hm_metric = RougeEntailmentHMComparison(rouge_metric, entailment_metric)
    exact_metric = ExactComparison()

    all_exact_exs = []
    all_hm_exs = []

    for graph in tqdm(graphs, desc='Duplicate metrics', total=len(graphs)):
        dupe_percentage, dupe_count_map = duplicate_percentage(graph)

        for k, v in dupe_count_map.items():
            count = duplicate_counts.get(k, 0)
            count += v
            duplicate_counts[k] = count

        percentage_of_duplicates += dupe_percentage

        #

        de_exact_dupes, ab_exact_dupes, de_exact_node_perc, ab_exact_node_perc, exact_exs = arg_comparison(graph, exact_metric, 0.9, track_examples)
        de_rehm_dupes, ab_rehm_dupes, de_rehm_node_perc, ab_rehm_node_perc, hm_exs = arg_comparison(graph, hm_metric, 0.7, track_examples)

        de_exact_step_arg_dupes.extend(de_exact_dupes)
        ab_exact_step_arg_dupes.extend(ab_exact_dupes)
        de_exact_step_node_dupes.extend(de_exact_node_perc)
        ab_exact_step_node_dups.extend(ab_exact_node_perc)

        de_rehm_step_arg_dupes.extend(de_rehm_dupes)
        ab_rehm_step_arg_dupes.extend(ab_rehm_dupes)
        de_rehm_step_node_dupes.extend(de_rehm_node_perc)
        ab_rehm_step_node_dupes.extend(ab_rehm_node_perc)

        all_exact_exs.extend(exact_exs)
        all_hm_exs.extend(hm_exs)

    percentage_of_duplicates /= len(graphs)

    fig, axs = plt.subplots(1)

    axs.bar(range(len(duplicate_counts)), list(duplicate_counts.values()), align='center')
    axs.set_xticks(range(len(duplicate_counts)), list(duplicate_counts.keys()))
    axs.set_title(f'Exact Duplicates ({percentage_of_duplicates*100:.2f}% are duplicates)')
    axs.set_xlabel("# Of Uses (1 is single use/non-duplicate, 2 and up means a duplicate)")
    axs.set_ylabel("Frequency")

    figfile = experiment_path / 'visualizations/exact_duplicates.png'
    if figfile.exists():
        os.remove(str(figfile))
    plt.savefig(str(figfile), dpi=200)


    fig, axs = plt.subplots(2)
    fig.subplots_adjust(hspace=.5)

    exact_data = {
        'Ded. Copied Arg': de_exact_step_arg_dupes,
        'Abd. Copied Arg': ab_exact_step_arg_dupes,
        'Ded. Dupes': de_exact_step_node_dupes,
        'Abd. Dupes': ab_exact_step_node_dups,
    }

    axs[0].boxplot(exact_data.values(), notch=True, patch_artist=True, boxprops=dict(facecolor="C0"))
    axs[0].set_xticklabels(list(exact_data.keys()))
    axs[0].set_title(f'Exact Duplicate Comparisons')
    axs[0].set_ylabel("Percentage")

    rehm_data = {
        'Ded. Copied Arg': de_rehm_step_arg_dupes,
        'Abd. Copied Arg': ab_rehm_step_arg_dupes,
        'Ded. Dupes': de_rehm_step_node_dupes,
        'Abd. Dupes': ab_rehm_step_node_dupes,
    }

    axs[1].boxplot(rehm_data.values(), notch=True, patch_artist=True, boxprops=dict(facecolor="C0"))
    axs[1].set_xticklabels(list(rehm_data.keys()))
    axs[1].set_title(f'Rouge+Entailment HM Duplicate Comparisons')
    axs[1].set_ylabel("Percentage")

    data = {
        'percentage_of_duplicates': percentage_of_duplicates,
        'duplicate_counts': duplicate_counts,
        'exact_data': exact_data,
        'rehm_data': rehm_data
    }

    figfile = experiment_path / 'visualizations/duplicates.png'
    if figfile.exists():
        os.remove(str(figfile))
    plt.savefig(str(figfile), dpi=200)

    datafile = experiment_path / 'visualizations/data/duplicates.json'
    if datafile.exists():
        os.remove(str(datafile))
    json.dump(data, datafile.open('w'))

    if track_examples:
        write_ex_file(experiment_path / 'visualizations/exact_dupes.csv', all_exact_exs)
        write_ex_file(experiment_path / 'visualizations/hm_dupes.csv', all_hm_exs)

def write_ex_file(file, data):
    if file.exists():
        os.remove(str(exact_dupes_file))
    with file.open('w') as f:
        writer = csv.DictWriter(f, list(data[0].keys()))
        writer.writerows(data)


def duplicate_percentage(graph: Graph):
    nodes: List[Node] = [*graph.premises, graph.goal]
    [nodes.extend(x.nodes) for x in [*graph.abductions, *graph.deductions]]

    dupe_gen_map: Dict[str, int] = {}
    count_map: Dict[int, int] = {}
    dupe_count = 0

    for node in nodes:
        count = dupe_gen_map.get(node.normalized_value, 0)
        count += 1
        dupe_gen_map[node.normalized_value] = count

    for v in dupe_gen_map.values():
        dupe_count += v - 1
        count = count_map.get(v, 0)
        count += 1
        count_map[v] = count

    return dupe_count / len(nodes), count_map


def arg_comparison(graph: Graph, metric: ComparisonMetric, threshold: float, track_examples: bool = False):
    deductive_argument_duplicates = []
    abductive_argument_duplicates = []

    ded_step_dupe_perc = []
    abd_step_dupe_perc = []

    examples = []

    for deduction in graph.deductions:
        vals = [x.normalized_value for x in deduction.nodes]
        scores = []
        for arg in deduction.arguments:
            scores.append(metric.score([graph[arg].normalized_value] * len(vals), vals))
            # deductive_argument_duplicates.append(sum([1 if x >= threshold else 0 for x in scores]))

        duped_arg_count = 0
        dupded_nodes = 0
        for nidx in range(len(vals)):
            nscores = sum([1 if scores[x][nidx] >= threshold else 0 for x in range(len(deduction.arguments))])
            duped_arg_count += nscores
            dupded_nodes += 1 if nscores > 0 else 0

            if not track_examples:
                continue

            for x in range(len(deduction.arguments)):
                s = scores[x][nidx]
                dupe = False
                if s > threshold:
                    dupe = True

                examples.append({'val': deduction[nidx].value, 'arg': graph[deduction.arguments[x]].value, 'score': s, 'dupe': dupe})


        deductive_argument_duplicates.append(duped_arg_count / max(1, len(vals) * 2))
        ded_step_dupe_perc.append(dupded_nodes / max(1, len(vals)))

    for abduction in graph.abductions:
        vals = [x.normalized_value for x in abduction.nodes]
        scores = []
        for arg in abduction.arguments:
            scores.append(metric.score([graph[arg].normalized_value] * len(vals), vals))
            # deductive_argument_duplicates.append(sum([1 if x >= threshold else 0 for x in scores]))

        duped_arg_count = 0
        dupded_nodes = 0
        for nidx in range(len(vals)):
            nscores = sum([1 if scores[x][nidx] >= threshold else 0 for x in range(len(abduction.arguments))])
            duped_arg_count += nscores
            dupded_nodes += 1 if nscores > 0 else 0

            if not track_examples:
                continue

            for x in range(len(abduction.arguments)):
                s = scores[x][nidx]
                dupe = False
                if s > threshold:
                    dupe = True

                examples.append(
                    {'val': abduction[nidx].value, 'arg': graph[abduction.arguments[x]].value, 'score': s, 'dupe': dupe})

        abductive_argument_duplicates.append(duped_arg_count / max(1, len(vals) * 2))
        abd_step_dupe_perc.append(dupded_nodes / max(1, len(vals)))

    return deductive_argument_duplicates, abductive_argument_duplicates, ded_step_dupe_perc, abd_step_dupe_perc, examples


if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument('--experiment_name', '-en', type=str,
                           help='Name of experiment')

    args = argparser.parse_args()

    _experiment_path: Path = SEARCH_OUTPUT_FOLDER / args.experiment_name

    search_stat_duplicate_generations(
        experiment_path=_experiment_path
    )
