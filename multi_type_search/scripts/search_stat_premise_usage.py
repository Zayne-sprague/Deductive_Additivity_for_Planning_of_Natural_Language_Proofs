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


def search_stat_premise_usage(
        experiment_path: Path
):
    search_file = experiment_path / 'output/searched.json'
    graphs = [Graph.from_json(x) for x in json.load(search_file.open('r'))]

    deductive_premise_use_percentages: Dict[str, int] = {}
    abductive_premise_use_percentages: Dict[str, int] = {}
    deductive_premise_use_frequency: Dict[str, int] = {}
    abductive_premise_use_frequency: Dict[str, int] = {}

    for graph in graphs:
        deductive_premise_counts = premise_usage(graph.deductions)
        abductive_premise_counts = premise_usage(graph.abductions)

        total_premises = len(graph.premises)

        deductive_percent = int(len(deductive_premise_counts) / total_premises * 100)
        abductive_percent = int(len(abductive_premise_counts) / total_premises * 100)

        count = deductive_premise_use_percentages.get(str(deductive_percent), 0)
        count += 1
        deductive_premise_use_percentages[str(deductive_percent)] = count

        count = abductive_premise_use_percentages.get(str(abductive_percent), 0)
        count += 1
        abductive_premise_use_percentages[str(abductive_percent)] = count

        for v in deductive_premise_counts.values():
            count = deductive_premise_use_frequency.get(v, 0)
            count += 1
            deductive_premise_use_frequency[v] = count

        for v in abductive_premise_counts.values():
            count = abductive_premise_use_frequency.get(v, 0)
            count += 1
            abductive_premise_use_frequency[v] = count

    fig, axs = plt.subplots(2, 2)
    fig.suptitle("Premise Usage")

    deductive_premise_use_percentages = dict(sorted(deductive_premise_use_percentages.items(), key=lambda x: int(x[0]), reverse=True))
    abductive_premise_use_percentages = dict(sorted(abductive_premise_use_percentages.items(), key=lambda x: int(x[0]), reverse=True))
    deductive_premise_use_frequency = dict(sorted(deductive_premise_use_frequency.items(), key=lambda x: int(x[0]), reverse=True))
    abductive_premise_use_frequency = dict(sorted(abductive_premise_use_frequency.items(), key=lambda x: int(x[0]), reverse=True))

    axs[0, 0].bar(range(len(deductive_premise_use_percentages)), list(deductive_premise_use_percentages.values()), align='center')
    axs[0, 0].set_xticks(range(len(deductive_premise_use_percentages)), list(deductive_premise_use_percentages.keys()))
    axs[0, 0].set_title(f'Deductive Premise Coverage')
    axs[0, 0].set_xlabel("% of coverage")
    axs[0, 0].set_ylabel("# of graphs")

    axs[0, 1].bar(range(len(abductive_premise_use_percentages)), list(abductive_premise_use_percentages.values()), align='center')
    axs[0, 1].set_xticks(range(len(abductive_premise_use_percentages)), list(abductive_premise_use_percentages.keys()))
    axs[0, 1].set_title(f'Abductive Premise Coverage')
    axs[0, 1].set_xlabel("% of coverage")
    axs[0, 1].set_ylabel("# of graphs")

    axs[1, 0].bar(range(len(deductive_premise_use_frequency)), list(deductive_premise_use_frequency.values()), align='center')
    axs[1, 0].set_xticks(range(len(deductive_premise_use_frequency)), list(deductive_premise_use_frequency.keys()))
    axs[1, 0].set_title(f'Deductive Premise Freq')
    axs[1, 0].set_xlabel("# Of or uses")
    axs[1, 0].set_ylabel("# of graphs")

    axs[1, 1].bar(range(len(abductive_premise_use_frequency)), list(abductive_premise_use_frequency.values()), align='center')
    axs[1, 1].set_xticks(range(len(abductive_premise_use_frequency)), list(abductive_premise_use_frequency.keys()))
    axs[1, 1].set_title(f'Abductive Premise Freq')
    axs[1, 1].set_xlabel("# of uses")
    axs[1, 1].set_ylabel("# of graphs")

    plt.tight_layout()

    figfile = experiment_path / 'visualizations/premise_usage.png'
    if figfile.exists():
        os.remove(str(figfile))
    plt.savefig(str(figfile))

    data = {
        'deductive_premise_use_percentages': deductive_premise_use_percentages,
        'abductive_premise_use_percentages': abductive_premise_use_percentages,
        'deductive_premise_use_frequency': deductive_premise_use_frequency,
        'abductive_premise_use_frequency': abductive_premise_use_frequency
    }

    datafile = experiment_path / 'visualizations/data/premise_usage.json'
    if datafile.exists():
        os.remove(str(datafile))
    json.dump(data, datafile.open('w'))


def premise_usage(
        hypernodes: List[HyperNode]
):
    premise_count = {}

    for hypernode in hypernodes:
        for arg in hypernode.arguments:
            key, _, _ = decompose_index(arg)
            if key == GraphKeyTypes.PREMISE:
                count = premise_count.get(arg, 0)
                count += 1
                premise_count[arg] = count

    return premise_count


if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument('--experiment_name', '-en', type=str,
                           help='Name of experiment')

    args = argparser.parse_args()

    _experiment_path: Path = SEARCH_OUTPUT_FOLDER / args.experiment_name

    search_stat_premise_usage(
        experiment_path=_experiment_path
    )
