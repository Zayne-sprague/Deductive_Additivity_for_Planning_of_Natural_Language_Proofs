from argparse import ArgumentParser
from pathlib import Path
import json
from tqdm import tqdm
from typing import List, Dict, Union, Optional
from multiprocessing import Pool

from multi_type_search.utils.paths import SEARCH_DATA_FOLDER
from multi_type_search.search.graph import Node, HyperNode, Graph, GraphKeyTypes, compose_index
from multi_type_search.search.comparison_metric import ComparisonMetric


def __process_wrapper__(kwargs):
    """Multiprocess function helper that passes kwargs into the right function"""
    return compare_nodes_to_missing_premises(**kwargs)


def compare_nodes_to_missing_premises_from_file(
        input_file: Path,
        score_name: str,
        comparison_metric: Union[ComparisonMetric, Dict[str, any]],
        output_file: Optional[Path] = None,
        force_output: bool = False,
        allowed_graphkeytypes: List[Union[GraphKeyTypes, str]] = (GraphKeyTypes.ABDUCTIVE,),
        use_normalized_values: bool = False,
        torch_devices: List[str] = ('cpu',),
        silent: bool = False,
) -> List[Graph]:
    """
    Loads a file that contains a List of Graph objects with nodes to compare with the missing premises of the graph.
    Check compare_nodes_to_missing_premises for more information on what the comparison is (this function wraps it).

    This function will break up the file into partitions and separate the work out across multiple torch devices if more
    than one is given.

    :param input_file: Path to file with graphs to run comparisons on
    :param score_name: Name of the score to save under the nodes.scores attribute
    :param comparison_metric: The comparison metric that will be used to compare the node and the missing premises (Can
        also be the json configuration of a comparison metric)
    :param output_file: Path to the output file that the scored graph will be stored in
    :param force_output: Overwrite anything currently written in the output file path.
    :param allowed_graphkeytypes: The nodes that are allowed to be compared in the graph
    :param use_normalized_values: Use the normalized value of the node over the raw value
    :param torch_devices: List of devices to split the scoring process across
    :param silent: No log messages
    :return: List of Graph objects where some (or all) of the nodes have been compared with the goal and have had the
        score saved in their node.scores attribute
    """

    # VALIDATE ARGS
    assert input_file.is_file() and (str(input_file).endswith('.json') or str(input_file).endswith('.jsonl')), \
        'Please specify a correct path to a json file with a list of graphs to run the scoring process over.'
    assert output_file is None or not output_file.exists() or force_output, \
        'Please specify an empty file path for the output parameter -o OR specify the force flag -f'

    allowed_graphkeytypes = [GraphKeyTypes[x] if isinstance(x, str) else x for x in allowed_graphkeytypes]

    if isinstance(comparison_metric, ComparisonMetric):
        comparison_metric = comparison_metric.to_json_config()

    # LOAD UP TREES
    if str(input_file).endswith('.jsonl'):
        json_graphs = list(jsonlines.open(str(input_file), 'r'))
    else:
        json_graphs = json.load(input_file.open('r'))
    all_graphs = [Graph.from_json(t) for t in json_graphs]

    device_graphs = []

    graphs_per_device = len(all_graphs) / len(torch_devices)
    for i in range(len(torch_devices)):
        start_idx = int(min(i * graphs_per_device, len(all_graphs) - 1))
        end_idx = int(min((i + 1) * graphs_per_device, len(all_graphs)))

        if i == len(torch_devices) - 1:
            # If its the last device, always take all the trees up till the last index.
            end_idx = len(all_graphs)

        device_graphs.append(all_graphs[start_idx:end_idx])

    compare_nodes_to_missing_premises_calls = []

    for idx, (torch_device, graphs) in enumerate(zip(torch_devices, device_graphs)):

        call_args = {
            'graphs': graphs,
            'score_name': score_name,
            'comparison_metric': comparison_metric,
            'allowed_graphkeytypes': allowed_graphkeytypes,
            'use_normalized_values': use_normalized_values,
            'torch_device': torch_device,
            '__job_idx__': idx
        }

        compare_nodes_to_missing_premises_calls.append(call_args)

    if len(torch_devices) == 1:
        results = [compare_nodes_to_missing_premises(**compare_nodes_to_missing_premises_calls[0])]
    else:
        with Pool(len(torch_devices)) as p:
            results = p.map(__process_wrapper__, compare_nodes_to_missing_premises_calls)

    scored_graphs = []
    for result in results:
        scored_graphs.extend(result)

    if output_file is not None:
        with output_file.open('w') as f:
            json.dump([x.to_json() for x in scored_graphs], f)

    if not silent:
        print("Finished scoring")

    return scored_graphs


def compare_nodes_to_missing_premises(
        graphs: List[Graph],
        score_name: str,
        comparison_metric: Union[ComparisonMetric, Dict[str, any]],
        allowed_graphkeytypes: List[Union[GraphKeyTypes, str]] = (GraphKeyTypes.ABDUCTIVE,),
        use_normalized_values: bool = False,
        torch_device: str = None,
        __job_idx__: int = 0
) -> List[Graph]:
    """
    Given a list of graphs, compare each node with the goal and save it's score under each nodes node.score attribute.

    :param graphs: List of graphs to compare nodes on
    :param score_name: Name of the score to save under the nodes.scores attribute
    :param comparison_metric: The comparison metric that will be used to compare the node and the goal (Can also be the
        json configuration of a comparison metric)
    :param allowed_graphkeytypes: The nodes that are allowed to be compared in the graph
    :param use_normalized_values: Use the normalized value of the node over the raw value
    :param torch_device: The torch device to load the comparison metric onto (None will use whatever the metric is set
        to currently / default).
    :param __job_idx__: If this function was called via compare_with_goal_from_file or is being ran in a multiprocessed
        way, then this variable controls where to show the TQDM progress bar based on the job idx (can be thought of as
        what line to show the progress bars on)
    :return: List of Graph objects where some (or all) of the nodes have been compared with the missing premises and
        have had the score saved in their node.scores attribute
    """

    allowed_graphkeytypes = [GraphKeyTypes[x] if isinstance(x, str) else x for x in allowed_graphkeytypes]

    if isinstance(comparison_metric, dict):
        comparison_metric = ComparisonMetric.from_config('comparison_metric', comparison_metric, device=torch_device)
    elif torch_device is not None:
        comparison_metric = comparison_metric.to(torch_device)

    for graph in tqdm(
            graphs,
            desc='Comparing Graph Nodes with Missing Premises',
            total=len(graphs),
            leave=False,
            position=__job_idx__
    ):
        for midx, missing_premise in enumerate(graph.missing_premises):
            for allowed_node_type in allowed_graphkeytypes:
                nodes = graph[allowed_node_type, :]

                if len(nodes) == 0:
                    continue

                if isinstance(nodes[0], Node):
                    group = [x.normalized_value if use_normalized_values else x.value for x in nodes]
                    targets = [missing_premise.normalized_value if use_normalized_values else missing_premise.value] * len(group)
                    scores = comparison_metric.score(group, targets)

                    for node, score in zip(nodes, scores):
                        node.scores[f'{score_name}_{midx}'] = score

                elif isinstance(nodes[0], HyperNode):
                    for hypernode in nodes:
                        group = [x.normalized_value if use_normalized_values else x.value for x in hypernode.nodes]
                        targets = [missing_premise.normalized_value if use_normalized_values else missing_premise.value] * len(group)
                        scores = comparison_metric.score(group, targets)

                        for node, score in zip(hypernode.nodes, scores):
                            node.scores[f'{score_name}_{midx}'] = score

                else:
                    raise Exception("Unknown Node type in score call")

    return graphs
