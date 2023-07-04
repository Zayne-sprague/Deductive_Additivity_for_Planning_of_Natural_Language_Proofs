from typing import Tuple
from enum import Enum


class GraphKeyTypes(Enum):
    """Different names of the types of steps that can be represented in a Graph Object"""
    PREMISE = 'PREMISE'
    MISSING_PREMISE = 'MISSING_PREMISE'
    DEDUCTIVE = 'DEDUCTIVE'
    ABDUCTIVE = 'ABDUCTIVE'
    GOAL = 'GOAL'


# How we separate the different parts of a graph index
__graph_index_delimiter__ = ":"


def compose_index(key: GraphKeyTypes, index: int = None, sub_index: int = None) -> str:
    """
    Helper function that composes a graph index given the type of generation being searched for (GraphKeyType) the main
    index of the generation (for premises, it's the only index value, for HyperNodes -- its the index of the hypernode)
    and the subindex of the generation (premises have none, hypernodes is the index of the specific node being looked
    for).

    :param key: The type of generation specified in GraphKeyTypes.
    :param index: The main index of the generation being looked for (Premises is the only index | Hypernodes is the
        HyperNode index)
    :param sub_index: The subindex of the generation being looked for (Premises can be None | Hypernodes is the specific
        Node index)
    :return: A string index that can be used on a graph object.
    """

    if index is not None and sub_index is not None:
        return f'{key.value}{__graph_index_delimiter__}{index}{__graph_index_delimiter__}{sub_index}'
    if index is not None:
        return f'{key.value}{__graph_index_delimiter__}{index}'
    return f'{key.value}'


def decompose_index(index: str) -> Tuple[GraphKeyTypes, int, int]:
    """
    Given a string index for a graph, this will break it down into it's component parts (the GraphKeyType, main index,
    and sub-index).

    :param index: The Graph Index to break down.
    :return: A tuple of the GraphKeyType, main index (if exists), subindex (if exists)
    """

    parts = index.split(__graph_index_delimiter__)

    key = GraphKeyTypes[parts[0]]
    index = None
    sub_index = None

    if len(parts) > 1:
        index = int(parts[1])
    if len(parts) > 2:
        sub_index = int(parts[2])

    return key, index, sub_index


def index_comp(index: str, other_index: str) -> bool:
    """
    Compares two indices against each other.
    :param index: First index to compare
    :param other_index: Second index to compare
    :return:
    """

    key1, index1, sub_index1 = decompose_index(index)
    key2, index2, sub_index2 = decompose_index(other_index)
    return key1 == key2 and index1 == index2 and sub_index1 == sub_index2
