import copy
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple
import json
from jsonlines import jsonlines
from tqdm import tqdm

from multi_type_search.utils.paths import SEARCH_DATA_FOLDER
from multi_type_search.search.graph import Node, HyperNode, Graph, GraphKeyTypes, compose_index


def create_shallow_graphs(
        input_file: Path,
        output_file: Path = None,
        force_output: bool = False,
        depth: int = 2,
        min_depth: int = 2,
        max_depth: int = 2,
        keep_extra_premises: bool = False,
        canonicalize: bool = True,
        silent: bool = False,
) -> List[Graph]:
    """
    Given a file containing a list of trees, convert those trees into a specific depth.

    :param input_file: Path to file with graphs to search over
    :param output_file: Path to the output file that the sliced graphs will be stored in
    :param force_output: Overwrite anything currently written in the output file path.
    :param depth: Depth to make the new graphs. (default is 2)
    :param min_depth: Only allow graphs greater than or equal to this value
    :param max_depth: Only allow graphs lesser than or equal to this value
    :param keep_extra_premises: Keep any extra premises not used in the shallow graph (needed when doing distractors)
    :param canonicalize: Write out canonical json (support for earlier tree-like formats, cannot be used with distractors)
    :param silent: No log messages
    :return: A list of Graph objects that are of the specified depth
    """

    # VALIDATE ARGS
    assert input_file.is_file() and (str(input_file).endswith('.json') or str(input_file).endswith('.jsonl')), \
        'Please specify a correct path to a json file with an array of trees to run the search over.'
    assert output_file is None or not output_file.exists() or force_output, \
        'Please specify an empty file path for the output parameter -o OR specify the force flag -f'

    # LOAD UP TREES
    if str(input_file).endswith('.jsonl'):
        json_graphs = list(jsonlines.open(str(input_file), 'r'))
    else:
        json_graphs = json.load(input_file.open('r'))
    graphs = [Graph.from_json(t) for t in json_graphs]

    shallow_graphs: List[Graph] = []

    # Slice trees.
    for graph in tqdm(graphs, desc='Slicing Graphs', total=len(graphs), position=0, leave=False):
        #pure_graph = copy.deepcopy(graph).slice(compose_index(GraphKeyTypes.DEDUCTIVE, len(graph.deductions) - 1, 0))
        pure_graph = copy.deepcopy(graph)

        sub_graphs = []

        if depth > -1:
            for hypernode_idx in range(len(graph)):
                node_idx = compose_index(GraphKeyTypes.DEDUCTIVE, hypernode_idx, 0)
                sub_graph = graph.slice(node_idx, depth)
                sub_graphs.append(sub_graph)
        else:
            sub_graphs.append(graph)

        allowed_sub_graphs = []

        for sub_graph in sub_graphs:
            if depth == -1 and min_depth == -1 and max_depth == -1:
                allowed_sub_graphs.append(sub_graph)
                continue

            graph_depth = sub_graph.get_depth()

            depth_check = depth == -1 or graph_depth == depth
            min_depth_check = min_depth == -1 or graph_depth >= min_depth
            max_depth_check = max_depth == -1 or graph_depth <= max_depth

            if depth_check and min_depth_check and max_depth_check:
                allowed_sub_graphs.append(sub_graph)

        if keep_extra_premises:
            for sub_graph in allowed_sub_graphs:
                sub_graph.distractor_premises = [x for x in [*graph.premises, *graph.distractor_premises] if x not in sub_graph.premises and x not in pure_graph.premises]

        shallow_graphs.extend(allowed_sub_graphs)

    if output_file is not None:
        # Save Sliced Trees.
        with output_file.open('w') as f:
            json.dump([x.to_canonical_json() if canonicalize else x.to_json() for x in shallow_graphs], f)

    if not silent:
        print(f"{len(shallow_graphs)} shallow trees created and saved in {output_file}.  Exiting...")

    return shallow_graphs


if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument('--input_file', '-i', type=str, help='Path to file with graphs to search over')
    argparser.add_argument('--output_file', '-o', type=str,
                           help='Path to the output file that the new graphs will be stored in', required=True)
    argparser.add_argument('--force_output', '-f', dest='force_output', action='store_true',
                           help='Overwrite anything currently written in the output file path.')
    argparser.add_argument('--depth', '-d', type=int,
                           help='Depth to make the new graphs. (default is 2)',
                           default=2)
    argparser.add_argument('--min_depth', '-md', type=int,
                           help='Only allow graphs greater than or equal to this value',
                           default=2)
    argparser.add_argument('--max_depth', '-mxd', type=int,
                           help='Only allow graphs less than or equal to this value',
                           default=2)
    argparser.add_argument('--keep_extra_premises', '-kep', dest='keep_extra_premises', action='store_true',
                           help='Although not used in the abductions and deductions, keep extra premises from the '
                                'original graph. (Needed for distractors)')
    argparser.add_argument('--canonicalize', '-c', dest='canonicalize', action='store_true',
                           help=' Write out canonical json (support for earlier tree-like formats, cannot be used with '
                                'distractors)')

    args = argparser.parse_args()

    _input_file: Path = Path(args.input_file)
    _output_file: Path = Path(args.output_file)
    _force_output: bool = args.force_output
    _depth: int = args.depth
    _min_depth: int = args.min_depth
    _max_depth: int = args.max_depth
    _keep_extra_premises: bool = args.keep_extra_premises
    _canonicalize: bool = args.canonicalize

    create_shallow_graphs(
        input_file=_input_file,
        output_file=_output_file,
        force_output=_force_output,
        depth=_depth,
        min_depth=_min_depth,
        max_depth=_max_depth,
        keep_extra_premises=_keep_extra_premises,
        canonicalize=_canonicalize
    )
