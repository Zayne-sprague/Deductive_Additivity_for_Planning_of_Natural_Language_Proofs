from multi_type_search.search.graph import Node, HyperNode, HyperNodeTypes, Graph, GraphKeyTypes, decompose_index, compose_index
from multi_type_search.search.comparison_metric import RougeComparison, SelfRougeComparison
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


def search_stat_self_rouge(
        experiment_path: Path
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

    for graph in tqdm(graphs, desc="Scoring Self-Rouge on graphs", total=len(graphs)):
        dg_selfrouge, ag_selfrouge, dsg_selfrouge, asg_selfrouge, allg_selfrouge, dsa_selfrouge, asa_selfrouge, all_sa_selfrouge = graph_self_rouge_scores(graph)

        deductive_gen_scores.extend(dg_selfrouge)
        abductive_gen_scores.extend(ag_selfrouge)
        [deductive_step_scores.extend(x) for x in dsg_selfrouge]
        [abductive_step_scores.extend(x) for x in asg_selfrouge]
        all_gen_scores.extend(allg_selfrouge)
        deductive_argument_scores.extend(dsa_selfrouge)
        abductive_argument_scores.extend(asa_selfrouge)
        all_argument_scores.extend(all_sa_selfrouge)

    data = {
        'D Gens': deductive_gen_scores,
        'A Gens': abductive_gen_scores,
        'D Step': deductive_step_scores,
        'A Step': abductive_step_scores,
        'All Gen': all_gen_scores,
        'D Args': deductive_argument_scores,
        'A Args': abductive_argument_scores,
        'All Args': all_argument_scores
    }

    fig, ax = plt.subplots()

    ax.boxplot(data.values(), notch=True, patch_artist=True, boxprops=dict(facecolor="C0"))
    ax.set_xticklabels(list(data.keys()))

    ax.set_title(f'Self-Rouge of Generations in the Search')
    ax.set_xlabel("Type of generation group")
    ax.set_ylabel("Average SelfRouge")

    figfile = experiment_path / 'visualizations/self_rouge.png'
    if figfile.exists():
        os.remove(str(figfile))
    plt.savefig(str(figfile))

    datafile = experiment_path / 'visualizations/data/self_rouge.json'
    if datafile.exists():
        os.remove(str(datafile))
    json.dump(data, datafile.open('w'))


def graph_self_rouge_scores(
        graph: Graph
):
    deductive_generations = [x.normalized_value for y in graph.deductions for x in y.nodes]
    abductive_generations = [x.normalized_value for y in graph.abductions for x in y.nodes]
    deductive_step_generations = [[x.normalized_value for x in y] for y in graph.deductions]
    abductive_step_generations = [[x.normalized_value for x in y] for y in graph.abductions]
    all_generations = [*deductive_generations, *abductive_generations]
    deductive_step_arguments = [graph[x].normalized_value for y in graph.deductions for x in y.arguments]
    abductive_step_arguments = [graph[x].normalized_value for y in graph.abductions for x in y.arguments]
    all_step_arguments = [*deductive_step_arguments, *abductive_step_arguments]

    rouge_metric = RougeComparison(rouge_types=['rouge1'])
    selfrouge = SelfRougeComparison(rouge_metric)

    dg_selfrouge = selfrouge.score(deductive_generations, [])
    ag_selfrouge = selfrouge.score(abductive_generations, [])
    dsg_selfrouge = [selfrouge.score(x, []) for x in deductive_step_generations]
    asg_selfrouge = [selfrouge.score(x, []) for x in abductive_step_generations]
    allg_selfrouge = selfrouge.score(all_generations, [])
    dsa_selfrouge = selfrouge.score(deductive_step_arguments, [])
    asa_selfrouge = selfrouge.score(abductive_step_arguments, [])
    all_sa_selfrouge = selfrouge.score(all_step_arguments, [])

    return dg_selfrouge, ag_selfrouge, dsg_selfrouge, asg_selfrouge, allg_selfrouge, dsa_selfrouge, asa_selfrouge, \
           all_sa_selfrouge








if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument('--experiment_name', '-en', type=str,
                           help='Name of experiment')

    args = argparser.parse_args()

    _experiment_path: Path = SEARCH_OUTPUT_FOLDER / args.experiment_name

    search_stat_self_rouge(
        experiment_path=_experiment_path
    )
