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

def plot_expansions(
        experiment_paths: List[Path],
        names: List[str]
):
    x_axis = []

    for e, n in zip(experiment_paths, names):
        data_file = e / 'visualizations/data/expansions.json'
        d = json.load(data_file.open('r'))
        steps = list(sorted(d['Total Expansion #']))

        counts = {}
        for i in steps:
            if str(i) not in counts:
                counts[str(i)] = 0
            counts[str(i)] += 1

        cur_val = 0
        x_values = []
        for i in range(0, 50):
            if str(i) in counts:
                cur_val += counts[str(i)]

            x_values.append(cur_val)
        x_axis.append(x_values)

    for (x, n) in zip(x_axis, names):
        plt.plot(list(range(0, 50)),x, label=n)

    plt.legend()
    plt.show()




if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument('--experiment_names', '-en', type=str, nargs="+",
                           help='Name of experiments')
    argparser.add_argument('--names', '-n', type=str, nargs="+",
                           help='Name for chart')

    args = argparser.parse_args()

    _experiment_paths: List[Path] = [SEARCH_OUTPUT_FOLDER / x for x in args.experiment_names]
    _names: List[str] = args.names

    plot_expansions(
        experiment_paths=_experiment_paths,
        names=_names
    )
