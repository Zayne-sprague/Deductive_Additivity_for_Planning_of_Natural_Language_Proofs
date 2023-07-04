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


def search_stat_expansions_until_proof(
        experiment_path: Path
):
    search_file = experiment_path / 'output/scored_comparisons.json'
    graphs = [Graph.from_json(x) for x in json.load(search_file.open('r'))]

    expansion_counts = []
    step_type_expansion_counts = []
    depth_counts = []
    all_times = []

    for graph in graphs:
        history_file = experiment_path / f'history/{graph.primitive_name}.jsonl'
        timing_file = experiment_path / f'timings/{graph.primitive_name}.json'

        total_expansion_count, step_type_idx, depth, time_to_first_proofs = expansion_count(graph, history_file, timing_file)
        if total_expansion_count == -1:
            continue

        expansion_counts.append(total_expansion_count)
        step_type_expansion_counts.append(step_type_idx)
        depth_counts.append(depth)
        all_times.append(time_to_first_proofs)

    data = {
        "Total Expansion #": expansion_counts,
        "S.T. Expansion #": step_type_expansion_counts,
        "Depth #": depth_counts,
        "Time to Proof": all_times
    }

    fig, ax = plt.subplots()

    ax.boxplot(data.values(), notch=True, patch_artist=True, boxprops=dict(facecolor="C0"))
    ax.set_xticklabels(list(data.keys()))

    ax.set_title(f'Expansion Metrics on Search')
    ax.set_xlabel("Expansion Metric")
    ax.set_ylabel("Average Count")

    figfile = experiment_path / 'visualizations/expansions.png'
    if figfile.exists():
        os.remove(str(figfile))
    plt.savefig(str(figfile))

    datafile = experiment_path / 'visualizations/data/expansions.json'
    if datafile.exists():
        os.remove(str(datafile))
    json.dump(data, datafile.open('w'))




def expansion_count(
        graph: Graph,
        history_file: Path,
        timing_file: Path
):
    if not history_file.exists():
        return -1, -1, -1, -1
    history = list(jsonlines.open(str(history_file), 'r'))
    timing_data = json.load(timing_file.open('r'))

    total_steps = 0
    abd_idx = 0
    ded_idx = 0
    proof_step = None
    proof_idx = None

    for line in history:
        total_steps += 1

        step_taken = line['step_taken']
        step_type = step_taken['step_type']

        generations = line['new_generations']
        abductions = generations['abductions']
        deductions = generations['deductions']

        if len(abductions) + len(deductions) == 0:
            continue

        if step_type == StepTypes.Deductive.value:
            step = graph.deductions[ded_idx]
            step_idx = compose_index(GraphKeyTypes.DEDUCTIVE, ded_idx)
            ded_idx += 1
        elif step_type == StepTypes.Abductive.value:
            step = graph.abductions[abd_idx]
            step_idx = compose_index(GraphKeyTypes.ABDUCTIVE, abd_idx)
            abd_idx += 1
        else:
            raise Exception("Unknown step type taken")

        tags = [x.tags.get('provided_proof') for x in step.nodes]
        if any(tags):
            node_provided_proof = tags.index(True)
            proof_step = step
            key, midx, _ = decompose_index(step_idx)
            proof_idx = compose_index(key, midx, node_provided_proof)
            break

    if proof_step is None:
        return -1, -1, -1, -1

    depth = graph.get_depth(proof_idx)
    _, step_type_idx, _ = decompose_index(proof_idx)

    time_to_first_proofs = timing_data['cumulative_step_times'][total_steps]
    return total_steps, step_type_idx+1, depth, time_to_first_proofs


if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument('--experiment_name', '-en', type=str,
                           help='Name of experiment')

    args = argparser.parse_args()

    _experiment_path: Path = SEARCH_OUTPUT_FOLDER / args.experiment_name

    search_stat_expansions_until_proof(
        experiment_path=_experiment_path
    )
