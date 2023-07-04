from multi_type_search.search.graph import Node, HyperNode, HyperNodeTypes, Graph, GraphKeyTypes, decompose_index, compose_index
from multi_type_search.search.comparison_metric import SelfBleuComparison
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


def search_stat_proofs(
        experiment_path: Path,
):
    search_file = experiment_path / 'output/proofs.json'
    all_proofs = [[Graph.from_json(x) for x in y] for y in json.load(search_file.open('r'))]

    proof_counts = []
    proofs_found = 0
    for graph_proofs in all_proofs:
        count = len(graph_proofs)
        if count > 0:
            proof_counts.append(count)
            proofs_found += 1

    data = {
        'Proof Counts': proof_counts
    }

    fig, ax = plt.subplots()

    ax.boxplot(data.values(), notch=True, patch_artist=True, boxprops=dict(facecolor="C0"))
    ax.set_xticklabels(list(data.keys()))

    ax.set_title(f'Proofs Per Graph (Coverage {proofs_found}/{len(all_proofs)})')

    data['proofs_found'] = proofs_found
    data['total'] = len(all_proofs)

    figfile = experiment_path / 'visualizations/proofs.png'
    if figfile.exists():
        os.remove(str(figfile))
    plt.savefig(str(figfile))

    datafile = experiment_path / 'visualizations/data/proofs.json'
    if datafile.exists():
        os.remove(str(datafile))
    json.dump(data, datafile.open('w'))


if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument('--experiment_name', '-en', type=str,
                           help='Name of experiment')

    args = argparser.parse_args()

    _experiment_path: Path = SEARCH_OUTPUT_FOLDER / args.experiment_name

    search_stat_proofs(
        experiment_path=_experiment_path
    )
