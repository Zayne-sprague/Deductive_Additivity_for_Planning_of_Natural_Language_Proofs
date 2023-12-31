from multi_type_search.search.graph import Node, GraphKeyTypes, compose_index

from enum import Enum
from typing import List, Dict, Optional


class HyperNodeTypes(Enum):
    """The valid types of a HyperNode."""
    Deductive = 'Deductive'
    Abductive = 'Abductive'
    Unknown = 'Unknown'


class HyperNode:
    """
    A primitive data structure for the Graph Object. Any set of Nodes can be combined to generate a "Step" and any
    "Step" can be oversampled to create numerous generations.  However, because these oversampled generations all share
    the same original argument Nodes, they are grouped into one HyperNode. A HyperNode is supposed to behave similar to
    a standard hypernode in graph-theory.
    """

    nodes: List[Node]
    arguments: List[str]
    type: HyperNodeTypes
    tags: Dict[str, any]

    def __init__(
            self,
            hypernode_type: HyperNodeTypes,
            nodes: List[Node],
            arguments: List[str],
            tags: Optional[Dict[str, any]] = None
    ):
        """
        :param hypernode_type: The type of the HyperNode being made (defined in HyperNodeTypes)
        :param nodes: The specific Nodes/generations that were generated by the HyperNode
        :param arguments: The graph indices of the arguments that created the Nodes/generations for the HyperNode.
        """

        self.type = hypernode_type
        self.nodes = nodes
        self.arguments = arguments
        self.tags = tags if tags is not None else {}

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, item) -> Node:
        return self.nodes[item]

    def __eq__(self, other) -> bool:
        """Equality check (mostly the arguments must match as well as the Nodes/generations)"""
        return isinstance(other, HyperNode)\
               and self.arguments is not None and other.arguments is not None \
               and len(self.arguments) == len(other.arguments) \
               and all([self.arguments[x] == other.arguments[x] for x in range(len(self.arguments))]) \
               and len(self.nodes) == len(other.nodes) \
               and all([self.nodes[x] == other.nodes[x] for x in range(len(self.nodes))])

    def __repr__(self) -> str:
        out = f'<HyperNode: {self.type.value} | # Nodes {len(self.nodes)}'
        if len(self.nodes) > 0:
            out += f' | [{self.nodes[0]}'

            if len(self.nodes) > 1:
                out += f', ..., {self.nodes[-1]}'

            out += ']'

        out += '>'

        return out

    def to_json(self) -> Dict[str, any]:
        """Function that converts a HyperNode into a json structure."""
        return {
            'type': self.type.value,
            'nodes': [x.to_json() for x in self.nodes],
            'arguments': self.arguments,
            'tags': self.tags if self.tags is not None else {}
        }

    def to_canonical_json(self) -> List[Dict[str, any]]:
        """
        Turns a HyperNode into a canonical format

        However, there were no HyperNodes in the canonical format, so instead this turns each of the HyperNodes nodes
        into their canonical format and returns that list of json.  It's up to the callee to appropriately handle them.
        """

        canonical_nodes = [x.to_canonical_json() for x in self.nodes]
        return canonical_nodes

    @classmethod
    def from_json(cls, json: Dict[str, any]) -> 'HyperNode':
        """
        Loads a HyperNode from it's json structure

        :param json: The json structure of the HyperNode
        :return: The newly instantiated HyperNode
        """

        return cls(
            hypernode_type=HyperNodeTypes[json.get("type", HyperNodeTypes.Unknown)],
            nodes=[Node.from_json(x) for x in json.get("nodes")],
            arguments=json.get('arguments'),
            tags=json.get('tags')
        )

    @classmethod
    def from_canonical_json(cls, json: Dict[str, any]) -> 'HyperNode':
        """
        Converts the canonical JSON format into a HyperNode.

        However there were no HyperNodes in the canonical version, so instead really what is given is a json object
        and that json object specifies a specific Node which is then wrapped around a HyperNode.

        :param json: The canonical Node in json structure that we want to wrap around in the new HyperNode
        :return: The newly instantiated HyperNode
        """

        def canonical_input_to_arg(inp: str) -> str:
            if 'p' in inp:
                return compose_index(GraphKeyTypes.PREMISE, int(inp[1:]))
            elif 'i' in inp:
                return compose_index(GraphKeyTypes.DEDUCTIVE, int(inp[1:]), 0)
            elif 'h' in inp:
                return compose_index(GraphKeyTypes.GOAL)

        return cls(
            hypernode_type=HyperNodeTypes.Unknown,
            nodes=[Node.from_json(json)],
            arguments=[canonical_input_to_arg(x) for x in json.get('inputs')]
        )

    def index(self, node: Node) -> int:
        """
        Function that finds the index of a specific node (if it exists, -1 otherwise).

        :param node: The node being searched for.
        :return: The index of the node in the HyperNode or -1 if it does not exist
        """

        for idx, other_node in enumerate(self.nodes):
            if node == other_node:
                return idx

        return -1

    def hypernode_compare(self, other_hypernode: 'HyperNode') -> bool:
        """
        Compares two HyperNodes on their type and arguments (but not their individual nodes)

        :param other_hypernode: HyperNode to compare against
        :return: A boolean which is True if they are the same or False if they are different.
        """

        return self.type == other_hypernode.type and all([x in self.arguments for x in other_hypernode.arguments])
