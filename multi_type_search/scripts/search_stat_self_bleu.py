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


def search_stat_self_bleu(
        experiment_path: Path,
        weights: List[float] = (0.33, 0.33, 0.33)
):
    search_file = experiment_path / 'output/searched.json'
    graphs = [Graph.from_json(x) for x in json.load(search_file.open('r'))]

    deductive_gen_scores = []
    abductive_gen_scores = []
    deductive_step_scores = []
    abductive_step_scores = []
    all_gen_scores = []
    deductive_argument_scores = []
    abductive_argument_scores = []
    all_argument_scores = []

    for graph in tqdm(graphs, desc="Scoring Self-Bleu on graphs", total=len(graphs)):
        dg, ag, dsg, asg, allg, dsa, asa, all_sa = graph_self_bleu_scores(graph, weights=weights)

        deductive_gen_scores.extend(dg)
        abductive_gen_scores.extend(ag)
        [deductive_step_scores.extend(x) for x in dsg]
        [abductive_step_scores.extend(x) for x in asg]
        all_gen_scores.extend(allg)
        deductive_argument_scores.extend(dsa)
        abductive_argument_scores.extend(asa)
        all_argument_scores.extend(all_sa)

    data = {
        'D Gens': deductive_gen_scores,
        'A Gens': abductive_gen_scores,
        'D Step': deductive_step_scores,
        'A Step': abductive_step_scores,
        'All Gen': all_gen_scores,
        'D Args': deductive_argument_scores,
        'A Args': abductive_argument_scores,
        'All Args': all_argument_scores,
    }

    fig, ax = plt.subplots()

    ax.boxplot(data.values(), notch=True, patch_artist=True, boxprops=dict(facecolor="C0"))
    ax.set_xticklabels(list(data.keys()))

    ax.set_title(f'Self-Bleu of Generations | W = ({", ".join([f"{x:.2f}" for x in weights])})')
    ax.set_xlabel("Type of generation group")
    ax.set_ylabel("Average Self-Bleu")

    figfile = experiment_path / 'visualizations/self_bleu.png'
    if figfile.exists():
        os.remove(str(figfile))
    plt.savefig(str(figfile))

    data['weights'] = weights

    datafile = experiment_path / 'visualizations/data/self_bleu.json'
    if datafile.exists():
        os.remove(str(datafile))
    json.dump(data, datafile.open('w'))


def graph_self_bleu_scores(
        graph: Graph,
        weights: List[float] = (.33, .33, .33)
):
    deductive_generations = [x.normalized_value for y in graph.deductions for x in y.nodes]
    abductive_generations = [x.normalized_value for y in graph.abductions for x in y.nodes]
    deductive_step_generations = [[x.normalized_value for x in y] for y in graph.deductions]
    abductive_step_generations = [[x.normalized_value for x in y] for y in graph.abductions]
    all_generations = [*deductive_generations, *abductive_generations]
    deductive_step_arguments = [graph[x].normalized_value for y in graph.deductions for x in y.arguments]
    abductive_step_arguments = [graph[x].normalized_value for y in graph.abductions for x in y.arguments]
    all_step_arguments = [*deductive_step_arguments, *abductive_step_arguments]

    metric = SelfBleuComparison()

    dg = metric.score(deductive_generations, [], weights=weights)
    ag = metric.score(abductive_generations, [], weights=weights)
    dsg = [metric.score(x, [], weights=weights) for x in deductive_step_generations]
    asg = [metric.score(x, [], weights=weights) for x in abductive_step_generations]
    allg = metric.score(all_generations, [], weights=weights)
    dsa = metric.score(deductive_step_arguments, [], weights=weights)
    asa = metric.score(abductive_step_arguments, [], weights=weights)
    all_sa = metric.score(all_step_arguments, [], weights=weights)

    return dg, ag, dsg, asg, allg, dsa, asa, all_sa


if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument('--experiment_name', '-en', type=str,
                           help='Name of experiment')

    args = argparser.parse_args()

    _experiment_path: Path = SEARCH_OUTPUT_FOLDER / args.experiment_name

    search_stat_self_bleu(
        experiment_path=_experiment_path
    )
