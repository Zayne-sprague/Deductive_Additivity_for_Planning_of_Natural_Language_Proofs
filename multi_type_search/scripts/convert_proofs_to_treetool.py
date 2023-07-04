from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple
import json
from jsonlines import jsonlines
from tqdm import tqdm
import csv
import random
random.seed(1)

from multi_type_search.utils.paths import SEARCH_DATA_FOLDER
from multi_type_search.search.graph import Graph
from multi_type_search.search.graph import HyperNode, Node
from multi_type_search.utils.paths import SEARCH_OUTPUT_FOLDER


def convert_proofs_to_treetool(
        input_files: List[Path] = (),
        input_experiments: List[Path] = (),
        output_file: Path = None,
        force_output: bool = False,
        first_k: int = None
):
    """
    Wrapper that can export a list of trees from a file or a list of trees where each tree is a list of proofs for that
    tree.

    :param input_files: Paths to files with trees to search over
    :param input_experiments: Paths to experiment folders with searches
    :param output_file: Path to the output file that the evaluations will be stored in
    :param force_output: Overwrite anything currently written in the output file path.
    """

    # VALIDATE ARGS
    assert (not output_file.exists() or force_output) and str(output_file).endswith('.json'), \
        'Please specify an empty csv file path for the output parameter -o OR specify the force flag -f'

    tree_sets = []
    file_names = []

    for input_file in input_files:
        # LOAD UP GRAPHS
        if str(input_file).endswith('.jsonl'):
            tree_sets.append(list(jsonlines.open(str(input_file), 'r')))
            file_names.append(input_file)
        else:
            tree_sets.append(json.load(input_file.open('r')))
            file_names.append(input_file)

    for experiment in input_experiments:
        scored_searches = experiment / 'output'

        for file in scored_searches.glob('proofs.json'):
            tree_sets.append(json.load(file.open('r')))
            file_names.append(file)

    if len(tree_sets) == 0:
        return

    for idx, tree_set in enumerate(tree_sets):
        top_proofs = []
        if first_k:
            tree_set = [x[0:min(len(x), first_k)] for x in tree_set]

        [top_proofs.extend(x) for x in tree_set]

        for pidx, proof in enumerate(top_proofs):
            proof = Graph.from_json(proof)
            for intermediate in proof.deductions:
                step_type = intermediate.tags.get('step_type')
                if not step_type:
                    step_type = 'abductive'
                elif step_type == 'bridge':
                    step_type = 'abductive'
                intermediate.tags['step_type'] = step_type
                for node in intermediate.nodes:
                    node.value = f'{node.value} | gs = {node.scores.get("goal_score", 0.):.2f}'
                    if node.scores.get('contrastive_d_score'):
                        node.value = f'{node.value} | ds = {node.scores.get("contrastive_d_score", 0.):.2f}'

            top_proofs[pidx] = proof.to_json()

        if not output_file:
            print(top_proofs)
        else:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with output_file.open('w') as f:
                json.dump(top_proofs, f)

if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument('--input_files', '-i', type=str, nargs='+', default=[],
                           help='Path to file with trees to search over')
    argparser.add_argument('--input_experiments', '-e', type=str, nargs='+',
                           help='Path to experiment folders that contain the searched trees.')
    argparser.add_argument('--output_file', '-o', type=str,
                           help='Path to the output file that the evaluations will be stored in', required=True)
    argparser.add_argument('--force_output', '-f', dest='force_output', action='store_true',
                           help='Overwrite anything currently written in the output file path.')
    argparser.add_argument('--first_k', '-k', type=int, help='Take the first k proofs.')

    args = argparser.parse_args()

    _input_files: List[Path] = [Path(x) for x in args.input_files]
    _input_experiments: List[Path] = [SEARCH_OUTPUT_FOLDER / x for x in args.input_experiments]
    _output_file: Path = Path(args.output_file)
    _force_output: bool = args.force_output
    _first_k: int = args.first_k

    convert_proofs_to_treetool(
        input_files=_input_files,
        input_experiments=_input_experiments,
        output_file=_output_file,
        force_output=_force_output,
        first_k=_first_k
    )
