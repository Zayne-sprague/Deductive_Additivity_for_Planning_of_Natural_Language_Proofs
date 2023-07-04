from multi_type_search.search.graph import Node, HyperNode, Graph, GraphKeyTypes, compose_index

from typing import List, Union, Optional, Tuple
from pathlib import Path
import json
from tqdm import tqdm


def find_proofs_from_file(
        input_file: Path,
        score_name: str,
        threshold: float,
        allowed_graphkeytypes: List[Union[GraphKeyTypes, str]] = (GraphKeyTypes.DEDUCTIVE,),
        output_file: Optional[Path] = None,
        updated_search_file: Optional[Path] = None,
        force_output: bool = False,
) -> List[List[Graph]]:
    # VALIDATE ARGS
    assert input_file.is_file() and (str(input_file).endswith('.json') or str(input_file).endswith('.jsonl')), \
        'Please specify a correct path to a json file with a list of graphs to find proofs in.'
    assert output_file is None or not output_file.exists() or force_output, \
        'Please specify an empty file path for the output parameter -o OR specify the force flag -f'

    allowed_graphkeytypes = [GraphKeyTypes[x] if isinstance(x, str) else x for x in allowed_graphkeytypes]

    # LOAD UP TREES
    if str(input_file).endswith('.jsonl'):
        json_graphs = list(jsonlines.open(str(input_file), 'r'))
    else:
        json_graphs = json.load(input_file.open('r'))

    all_graphs = [Graph.from_json(t) for t in json_graphs]

    proofs, tagged_graphs = find_proofs(
        all_graphs,
        score_name=score_name,
        threshold=threshold,
        allowed_graphkeytypes=allowed_graphkeytypes
    )

    if output_file is not None:
        with output_file.open('w') as f:
            json.dump([[y.to_json() for y in x] for x in proofs], f)

    if updated_search_file is not None:
        with updated_search_file.open('w') as f:
            json.dump([x.to_json() for x in tagged_graphs], f)

    return proofs


def find_proofs(
        graphs: List[Graph],
        score_name: str,
        threshold: float,
        allowed_graphkeytypes: List[Union[GraphKeyTypes, str]] = (GraphKeyTypes.DEDUCTIVE,),
) -> Tuple[List[List[Graph]], List[Graph]]:
    """
    This function returns a list of proofs given a set of graphs with deductions and abductions in it.  Each of those
    HyperNodes should have a score under node.scores[{score_name}] that if above the given threshold, is considered as
    having "solved" the proof or otherwise can be used to build a proof from the premises of the graph to the goal.

    :param graphs: Graphs to find proofs in
    :param score_name: The name of the score to look for in the node.score attribute
    :param threshold: The value the score must be before being considered a proof.
    :param allowed_graphkeytypes: The allowed GraphKeyType to check for solving the proof
    :return: A list of proofs per graph that have only deductions and only the deductions that lead to the goal, as well
        as the original graphs (they will have their nodes updated with tags that provided a proof)
    """

    allowed_graphkeytypes = [GraphKeyTypes[x] if isinstance(x, str) else x for x in allowed_graphkeytypes]

    proofs: List[List[Graph]] = []

    for graph in tqdm(
            graphs,
            desc='Finding Proofs',
            total=len(graphs),
    ):
        graph_proofs = []

        for allowed_node_type in allowed_graphkeytypes:
            nodes = graph[allowed_node_type, :]

            if len(nodes) == 0:
                continue

            if isinstance(nodes, HyperNode):
                nodes = [nodes]

            if allowed_node_type == GraphKeyTypes.PREMISE:
                for nidx, node in enumerate(nodes):
                    score = node.scores.get(score_name, -1)
                    if score >= threshold:
                        index = compose_index(allowed_node_type, nidx)

                        graph[index].tags['provided_proof'] = True

                        graph_proofs.append(graph.reduce_to_deductions(index))

            elif isinstance(nodes[0], HyperNode):
                for hidx, hypernode in enumerate(nodes):
                    for nidx, node in enumerate(hypernode.nodes):
                        score = node.scores.get(score_name, -1)
                        if score >= threshold:
                            index = compose_index(allowed_node_type, hidx, nidx)

                            graph[index].tags['provided_proof'] = True

                            graph_proofs.append(graph.reduce_to_deductions(index))
            else:
                raise Exception("Unknown Node type in score call")

        graph_proofs = list(sorted(graph_proofs, key=lambda x: (x.deductions[-1].nodes[0].scores.get('contrastive_d_score', 0) + x.deductions[-1].nodes[0].scores.get('goal_score', 0)) / 2, reverse=True))
        proofs.append(graph_proofs)

    return proofs, graphs
