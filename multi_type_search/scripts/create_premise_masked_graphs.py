from argparse import ArgumentParser
from pathlib import Path
from typing import List
import json
from tqdm import tqdm
from copy import deepcopy
from itertools import combinations
import jsonlines

from multi_type_search.utils.paths import SEARCH_DATA_FOLDER
from multi_type_search.search.graph import Node, HyperNode, Graph, GraphKeyTypes, compose_index


def create_premise_masked_graphs(
        input_file: Path,
        output_file: Path = None,
        force_output: bool = False,
        premises_to_mask: int = 1,
        silent: bool = False
) -> List[Graph]:
    """
    Given a file which contains a list of trees, create variants of each tree where one of the premises is masked.

    :param input_file: Path to file with graphs to search over
    :param output_file: Path to the output file that the updated graphs will be stored in
    :param force_output: Overwrite anything currently written in the output file path.
    :param premises_to_mask: How many premises to mask (default is 1)
    :param silent: No log messages
    :return: List of Graph objects with a premise masked.
    """

    # VALIDATE ARGS
    assert input_file.is_file() and (str(input_file).endswith('.json') or str(input_file).endswith('.jsonl')), \
        'Please specify a correct path to a json file with an array of graphs to run the search over.'
    assert output_file is None or not output_file.exists() or force_output, \
        'Please specify an empty file path for the output parameter -o OR specify the force flag -f'

    # LOAD UP TREES
    if str(input_file).endswith('.jsonl'):
        json_graphs = list(jsonlines.open(str(input_file), 'r'))
    else:
        json_graphs = json.load(input_file.open('r'))

    graphs = [Graph.from_json(t) for t in json_graphs]

    masked_graphs: List[Graph] = []

    # Slice trees.
    for graph in tqdm(graphs, desc='Masking Trees', total=len(graphs), position=0, leave=False):
        premises = list(range(len(graph.premises)))

        if premises_to_mask == 0:
            masked_graphs.append(graph)
            continue

        masks = combinations(premises, min([len(premises), premises_to_mask]))

        # For each mask, mask a unique premise we haven't masked before and store it as an example to run
        for mask in masks:
            masked_example = deepcopy(graph)

            # For each premise idx, mask the associated premise.  If we sort, we do not have to worry about premise
            # re-indexing (i.e. masking premise 1 will make all premises above it their idx minus 1.
            for premise_idx in sorted(mask, reverse=True):
                masked_example.mask(compose_index(GraphKeyTypes.PREMISE, premise_idx))

            masked_graphs.append(masked_example)

    if output_file is not None:
        # Save Sliced Trees.
        with output_file.open('w') as f:
            json.dump([x.to_json() for x in masked_graphs], f)

    if not silent:
        print(f"{len(masked_graphs)} masked trees created.")

        if output_file is not None:
            print(f'Masked graphs saved in {output_file}.')

    return masked_graphs


if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument('--input_file', '-i', type=str, help='Path to file with graphs to search over')
    argparser.add_argument('--output_file', '-o', type=str,
                           help='Path to the output file that the updated graphs will be stored in')
    argparser.add_argument('--force_output', '-f', dest='force_output', action='store_true',
                           help='Overwrite anything currently written in the output file path.')
    argparser.add_argument('--premises_to_mask', '-p', type=int, default=1,
                           help='How many premises to mask (default is 1)')

    args = argparser.parse_args()

    _input_file: Path = Path(args.input_file)
    _output_file: Path = Path(args.output_file)
    _force_output: bool = args.force_output
    _premises_to_mask: int = args.premises_to_mask

    create_premise_masked_graphs(
        input_file=_input_file,
        output_file=_output_file,
        force_output=_force_output,
        premises_to_mask=_premises_to_mask
    )
