from multi_type_search.search.graph import Node, HyperNode, HyperNodeTypes, Graph, GraphKeyTypes, decompose_index, compose_index
from multi_type_search.search.step_type import StepTypes
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


def search_stat_timings(
        experiment_path: Path,
):
    timings_dir = experiment_path / 'timings'

    all_step_times = []
    all_search_times = []
    all_step_queue_population_counts = []
    all_total_steps_taken = []

    for file in timings_dir.glob("*.json"):
        with file.open('r') as f:
            timing_data = json.load(f)

        all_search_times.append(timing_data['total_time'])
        all_step_times.append(timing_data['step_times'])
        all_step_queue_population_counts.append(timing_data['step_queue_populations'])
        all_total_steps_taken.append(len(timing_data['step_times']))

    largest_step_times = max([len(x) for x in all_step_times])
    step_times_by_step = [[x[y] for x in all_step_times if len(x) > y] for y in range(largest_step_times)]
    avg_stxs = [statistics.mean(x) for x in step_times_by_step]

    fig, ax = plt.subplots()

    ax.plot(range(len(avg_stxs)), avg_stxs)

    ax2 = ax.twinx()

    ax2.plot(range(len(avg_stxs)), [len(x) for x in step_times_by_step], color='orange')

    plt.show()


if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument('--experiment_name', '-en', type=str,
                           help='Name of experiment')

    args = argparser.parse_args()

    _experiment_path: Path = SEARCH_OUTPUT_FOLDER / args.experiment_name

    search_stat_timings(
        experiment_path=_experiment_path
    )
