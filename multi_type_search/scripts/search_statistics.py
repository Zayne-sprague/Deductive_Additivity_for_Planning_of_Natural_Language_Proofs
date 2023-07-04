import shutil

from multi_type_search.search.graph import Node, HyperNode, HyperNodeTypes, Graph, GraphKeyTypes, decompose_index, compose_index
from multi_type_search.utils.paths import SEARCH_OUTPUT_FOLDER

from pathlib import Path
from typing import List, Dict, Tuple
import json
from jsonlines import jsonlines
from argparse import ArgumentParser
from copy import deepcopy
from tqdm import tqdm
import statistics
import matplotlib.pyplot as plt


def experiment_search_statistics(
        experiment_path: Path
):
    if Path('./tmp').exists():
        shutil.rmtree('./tmp')
    search_file = experiment_path / 'output/searched.json'
    history_directory = experiment_path / 'history'

    searched_graphs = [Graph.from_json(x) for x in json.load(search_file.open('r'))]

    return search_statistics(
        searched_graphs,
        history_directory
    )


def search_statistics(
        searched_graphs: List[Graph],
        history_directory: Path
):
    aligned_history_and_idx = match(searched_graphs, history_directory)
    #
    backup_file = Path('./tmp2.json')
    with backup_file.open('w') as f:
        json.dump(aligned_history_and_idx, f)

    aligned_history_and_idx = json.load(backup_file.open('r'))

    graph_and_history = [(searched_graphs[idx], Path(file)) for (idx, file) in aligned_history_and_idx]

    hypernode_stats = hypernode_hits(graph_and_history)

    step_by_step_graphs(graph_and_history)


def hypernode_hits(
        graph_and_history: List[Tuple[Graph, Path]]
):
    stats = {}
    all_abductive_counts = []
    all_deductive_counts = []

    all_abductive_depths = []
    all_deductive_depths = []

    for graph, history_file in graph_and_history:

        history = list(jsonlines.open(str(history_file), 'r'))

        graph_stats = {'Deductive': {0: 0}, 'Abductive': {0: 0}}
        depth_stats = {'Deductive': {0: 0}, 'Abductive': {0: 0}}

        for line in history:
            step_taken = line.get('step_taken')
            step_type = step_taken.get('step_type')
            arguments = step_taken.get('arguments')

            for arg in arguments:
                arg_key, hidx, _ = decompose_index(arg)

                depth = graph.get_depth(arg)
                step_type_depths = depth_stats.get(step_type, {})
                depth_count = step_type_depths.get(depth, 0)
                depth_count += 1
                step_type_depths[depth] = depth_count
                depth_stats[step_type] = step_type_depths

                if arg_key == GraphKeyTypes.PREMISE or arg_key == GraphKeyTypes.GOAL:
                    continue

                step_type_stats = graph_stats.get(step_type, {})

                hypernode = compose_index(arg_key, hidx)
                hypernode_class_stats = step_type_stats.get(hypernode, 0)
                hypernode_class_stats += 1

                step_type_stats[hypernode] = hypernode_class_stats
                graph_stats[step_type] = step_type_stats

        abduction_counts = [max([x for x in graph_stats['Abductive'].values()])]
        deduction_counts = [max([x for x in graph_stats['Deductive'].values()])]

        abduction_depths = [max([x for x in depth_stats['Abductive'].values()])]
        deduction_depths = [max([x for x in depth_stats['Deductive'].values()])]

        all_abductive_counts.extend(abduction_counts)
        all_deductive_counts.extend(deduction_counts)

        all_abductive_depths.extend(abduction_depths)
        all_deductive_depths.extend(deduction_depths)

    # deductive_mean = statistics.mean(all_deductive_counts)
    # abductive_mean = statistics.mean(all_abductive_counts)
    #
    # abduction_std = statistics.pstdev(all_abductive_counts)
    # deduction_std = statistics.pstdev(all_deductive_counts)

    plot_counts(all_abductive_counts, 'abductive_hypernode_count')
    plot_counts(all_deductive_counts, 'deductive_hypernode_count')

    plot_counts(all_abductive_depths, 'abductive_depths')
    plot_counts(all_deductive_depths, 'deductive_depths')
    return stats


def plot_counts(counts, title):
    a = {}
    for j in counts:
        v = a.get(j, 0)
        v += 1
        a[j] = v

    items = sorted(a.items())
    x = [j[0] for j in items]
    y = [j[1] for j in items]

    plt.plot(x, y)
    plt.title(title)
    plt.show()


def step_by_step_graphs(
        graph_and_history: List[Tuple[Graph, Path]]
):
    for graph, history_file in graph_and_history:

        history = list(jsonlines.open(str(history_file), 'r'))

        deductive_graphs: List[Graph] = []
        abductive_graphs: List[Graph] = []

        for line in history:
            step_taken = line.get('step_taken')
            step_type = step_taken.get('step_type')
            arguments = step_taken.get('arguments')
            score = step_taken.get('score')

            premises = []
            [premises.append(graph[x]) for x in arguments]
            premises.append(Node(
                f'INFO: {step_type}({arguments[0]}, {arguments[1]}) | '
                f'Depth = {1 + max(graph.get_depth(arguments[0]), graph.get_depth(arguments[1]))} | '
                f'Score = {score} | '
                f'FileName = {history_file.name.replace(".json", "")}'
            ))

            [premises.append(Node(f'ORIG_PREMISE: {x.value}')) for x in graph.__original_premises__]


            if step_type == 'Abductive':

                hypernodes = [HyperNode.from_json(x) for x in line.get('new_generations', {}).get('abductions')]
                for hypernode in hypernodes:
                    hypernode.arguments = [compose_index(GraphKeyTypes.PREMISE, 0), compose_index(GraphKeyTypes.PREMISE, 1)]

                new_graph = Graph(
                    goal=graph.goal,
                    premises=premises,
                    abductions=hypernodes
                )

                new_graph.missing_premises = graph.missing_premises

                abductive_graphs.append(new_graph)
            else:

                hypernodes = [HyperNode.from_json(x) for x in line.get('new_generations', {}).get('deductions')]
                for hypernode in hypernodes:
                    hypernode.arguments = [compose_index(GraphKeyTypes.PREMISE, 0), compose_index(GraphKeyTypes.PREMISE, 1)]

                new_graph = Graph(
                    goal=graph.goal,
                    premises=premises,
                    deductions=hypernodes
                )

                new_graph.missing_premises = graph.missing_premises

                deductive_graphs.append(new_graph)

        tmp_dir = Path(f'./tmp/{history_file.name.replace(".json", "")}')
        if tmp_dir.exists():
            shutil.rmtree(str(tmp_dir))
        tmp_dir.mkdir(exist_ok=True, parents=True)
        with (tmp_dir / 'deductions.json').open('w') as f:
            json.dump([x.to_json() for x in deductive_graphs], f)

        with (tmp_dir / 'abductions.json').open('w') as f:
            json.dump([x.to_json() for x in abductive_graphs], f)


def match(
        searched_graphs: List[Graph],
        history_directory: Path
):
    aligned_history_and_idx = []
    found_idxs = []

    files = list(history_directory.glob("*.jsonl"))
    total = len(files)

    for file in tqdm(files, total=total, desc='Matching'):
        history = list(jsonlines.open(str(file), 'r'))
        considered_graphs = deepcopy(list(zip(list(range(len(searched_graphs))), searched_graphs)))

        abductions = 0
        deductions = 0
        for line in history:
            step_taken = line.get('step_taken', {})
            step_type = step_taken.get('step_type')
            arguments = step_taken.get('arguments')

            produced_generation = False
            if step_type == 'Abductive':
                produced_generation = len(line.get('new_generations', {}).get('abductions', [])) > 0
            else:
                produced_generation = len(line.get('new_generations', {}).get('deductions', [])) > 0

            if not produced_generation:
                continue


            for cgidx, (idx, graph) in enumerate(considered_graphs):
                if idx in found_idxs:
                    considered_graphs.remove(considered_graphs[cgidx])
                    continue

                elif step_type == 'Abductive':
                    if len(graph.abductions) > abductions:
                        matching_hypernode = graph.abductions[abductions]
                    else:
                        considered_graphs.remove(considered_graphs[cgidx])
                        continue
                else:
                    if len(graph.deductions) > deductions:
                        matching_hypernode = graph.deductions[deductions]
                    else:
                        considered_graphs.remove(considered_graphs[cgidx])
                        continue

                if matching_hypernode.arguments != arguments:
                    considered_graphs.remove(considered_graphs[cgidx])

            if step_type == 'Abductive':
                abductions += 1
            else:
                deductions += 1
            if len(considered_graphs) == 1:
                aligned_history_and_idx.append((considered_graphs[0][0], str(file)))
                found_idxs.append((considered_graphs[0][0]))
                break
            elif len(considered_graphs) == 0:
                raise Exception("Wrong")
    return aligned_history_and_idx


if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument('--experiment_name', '-en', type=str,
                           help='Name of experiment')

    args = argparser.parse_args()

    _experiment_path: Path = SEARCH_OUTPUT_FOLDER / args.experiment_name

    experiment_search_statistics(
        experiment_path=_experiment_path
    )
