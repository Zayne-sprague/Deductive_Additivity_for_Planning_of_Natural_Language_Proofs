from multi_type_search.search.graph import Node, HyperNode, HyperNodeTypes, Graph, GraphKeyTypes, decompose_index, compose_index
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


def search_stat_basics(
        experiment_path: Path
):
    search_file = experiment_path / 'output/searched.json'
    graphs = [Graph.from_json(x) for x in json.load(search_file.open('r'))]

    all_deductive_depths = []
    all_abductive_depths = []

    all_deductive_arg_use_counts = []
    all_abductive_arg_use_counts = []

    for graph in graphs:
        deductive_depths, abductive_depths, deductive_argument_use_counts, abductive_argument_use_counts = basic_search_stats(graph)

        all_deductive_depths.extend(deductive_depths)
        all_abductive_depths.extend(abductive_depths)
        all_deductive_arg_use_counts.extend(deductive_argument_use_counts)
        all_abductive_arg_use_counts.extend(abductive_argument_use_counts)

    data = {
        "Ded Depth": all_deductive_depths,
        "Abd Depth": all_abductive_depths,
        "Ded Sample Rate": all_deductive_arg_use_counts,
        "Abd Sample Rate": all_abductive_arg_use_counts
    }

    fig, ax = plt.subplots()

    ax.boxplot(data.values(), notch=True, patch_artist=True, boxprops=dict(facecolor="C0"))
    ax.set_xticklabels(list(data.keys()))

    ax.set_title(f'Basic Search Stats')
    ax.set_xlabel("Stat")
    ax.set_ylabel("Count")

    figfile = experiment_path / 'visualizations/basic.png'
    if figfile.exists():
        os.remove(str(figfile))
    plt.savefig(str(figfile))

    datafile = experiment_path / 'visualizations/data/basic.json'
    if datafile.exists():
        os.remove(str(datafile))
    json.dump(data, datafile.open('w'))


def basic_search_stats(graph: Graph):
    deductive_depths = []
    abductive_depths = []

    deductive_argument_use_map = {}
    abductive_argument_use_map = {}

    for didx, deduction in enumerate(graph.deductions):
        idx = compose_index(GraphKeyTypes.DEDUCTIVE, didx)

        deductive_depths.append(graph.get_depth(idx))

        for arg in deduction.arguments:
            count = deductive_argument_use_map.get(arg, 0)
            count += 1
            deductive_argument_use_map[arg] = count

    for aidx, abduction in enumerate(graph.abductions):
        idx = compose_index(GraphKeyTypes.ABDUCTIVE, aidx)

        abductive_depths.append(graph.get_depth(idx))

        for arg in abduction.arguments:
            count = abductive_argument_use_map.get(arg, 0)
            count += 1
            abductive_argument_use_map[arg] = count

    deductive_argument_use_counts = list(deductive_argument_use_map.values())
    abductive_argument_use_counts = list(abductive_argument_use_map.values())

    return deductive_depths, abductive_depths, deductive_argument_use_counts, abductive_argument_use_counts


if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument('--experiment_name', '-en', type=str,
                           help='Name of experiment')

    args = argparser.parse_args()

    _experiment_path: Path = SEARCH_OUTPUT_FOLDER / args.experiment_name

    search_stat_basics(
        experiment_path=_experiment_path
    )
