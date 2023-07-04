from multi_type_search.search.graph import Node, HyperNode, HyperNodeTypes, GraphKeyTypes, compose_index, \
    decompose_index, index_comp
from multi_type_search.utils.search_utils import normalize

from typing import List, Dict, Iterable, Tuple, ClassVar, Union, Optional
from copy import deepcopy
from functools import partial
import sys
from enum import Enum
import uuid
import hashlib
import json


class Graph:
    """
    The Graph object is meant to be an extension of something akin to an Entailment Tree.  Where the graph contains a
    set of HyperNodes containing generated statements from a set of arguments, where the arguments can be other
    generations in a HyperNode or a Premise (a fact given to the graph and was not generated).

    Statements can be made through different types of inference (Deduction or Abduction) which extends this
    representation beyond a tree and into a graph (since they are effectively exploring the space of all generations
    from two directions and are often unconnected).
    """

    goal: Node
    premises: List[Node]
    abductions: List[HyperNode]
    deductions: List[HyperNode]

    missing_premises: List[Node]  # Intentionally removed or masked premises
    __original_premises__: List[Node]  # List of original premises (self.premises + self.missing_premises)

    # TODO - probably shouldn't be stored here.
    distractor_premises: List[Node]  # Storage for premises not really related to the graph.

    def __init__(
            self,
            goal: Union[Node, str],
            premises: Union[List[Node], List[str]] = (),
            abductions: List[HyperNode] = (),
            deductions: List[HyperNode] = ()
    ):
        """
        :param goal: The root of the tree that generations are attempting to lead to.
        :param premises: List of the premises for the graph (no particular order)
        :param abductions: Abductive Hypernodes for the graph
        :param deductions: Deductive Hypernodes for the graph
        """

        self.goal = Node(goal) if isinstance(goal, str) else goal

        self.premises = [Node(x) if isinstance(x, str) else x for x in premises]
        #self.__original_premises__ = deepcopy(self.premises)
        self.__original_premises__ = self.premises

        self.abductions = abductions or []
        self.deductions = deductions or []

        self.missing_premises = []
        self.distractor_premises = []

    @property
    def primitive_name(self) -> str:
        """
        This property stores the unique name of the graph given its goal and initial set of premises.  This allows
        for easily linking graphs together if you want to store them in a file format.

        This should be invariant across machines.
        :returns: A unique string id conditioned on the graphs Goal and initial Premises.
        """
        key = json.dumps(
            {
                'premises': [x.to_json() for x in self.__original_premises__],
                'goal': self.goal.to_json()
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=None,
            separators=(',', ':')
        )

        return hashlib.md5(key.encode('utf-8')).digest().hex()

    @classmethod
    def from_json(cls, graph_json: Dict[str, any]) -> 'Graph':
        """
        Construct a graph from a json structure.

        :param graph_json: The json structure of the graph
        :return: The newly instantiated Graph
        """

        # If this graph is in the canonical format, use that json loader.
        if 'goal' not in graph_json:
            return cls.from_canonical_json(graph_json)

        goal = Node.from_json(graph_json['goal'])
        premises = [Node.from_json(x) for x in graph_json['premises']]
        abductions = [HyperNode.from_json(x) for x in graph_json['abductions']]
        deductions = [HyperNode.from_json(x) for x in graph_json['deductions']]

        graph = cls(goal=goal, premises=premises, abductions=abductions, deductions=deductions)

        if 'missing_premises' in graph_json:
            graph.missing_premises = [Node.from_json(x) for x in graph_json['missing_premises']]
        if 'original_premises' in graph_json:
            graph.__original_premises__ = [Node.from_json(x) for x in graph_json['original_premises']]
        if 'distractor_premises' in graph_json:
            graph.distractor_premises = [Node.from_json(x) for x in graph_json['distractor_premises']]

        return graph

    @classmethod
    def from_canonical_json(cls, canonical_json: Dict[str, any]) -> 'Graph':
        """
        Loads a graph instance given the graphs canonical json format

        :param canonical_json: The json of the graph in canonical format
        :returns: THe newly instantiated Graph
        """

        def canonical_index_to_graph_index(
                index: str,
                intermediate_map: Dict[str, str],
                deductions_map: Dict[str, any],
        ):
            """Helper for turning a canonical index into a Graph Index"""
            if 'p' in index:
                return compose_index(GraphKeyTypes.PREMISE, int(index[1:]))
            if 'h' in index:
                return compose_index(GraphKeyTypes.GOAL)
            if 'i' in index:
                arg_key = intermediate_map[index]
                hypernode = deductions_map.get(arg_key, {})
                hidx = hypernode.get('hidx', -1)
                nidx_map = hypernode.get('nidx_map', {})
                nidx = nidx_map.get(index, -1)
                assert hidx != -1 and nidx != -1, 'Incorrect mapping between intermediates and deductions'
                return compose_index(GraphKeyTypes.DEDUCTIVE, hidx, nidx)

        goal = Node(value=canonical_json['hypothesis'])
        premises = [Node(x) for x in canonical_json['premises']]
        # deductions = [HyperNode.from_canonical_json(x) for x in canonical_json['intermediates']]

        hypernode_map = {}
        intermediate_map = {}
        hypernode_counter = 0
        idx = 0
        for intermediate in canonical_json['intermediates']:
            args = ",".join(intermediate['inputs'])

            hypernode = hypernode_map.get(args, None)
            if hypernode is None:
                hypernode = {'hidx': hypernode_counter, 'nodes': [], 'nidx_map': {}}
                hypernode_counter += 1

            if intermediate['output'] not in [x.value for x in hypernode['nodes']]:
                hypernode['nidx_map'][f'i{idx}'] = len(hypernode['nodes'])
                hypernode['nodes'].append(Node(intermediate['output']))
            else:
                # TODO - some annotations have same arguments multiple times producing the same output.  This is here to
                #    make these duplicate steps point to the same hypernode subnode.
                hypernode['nidx_map'][f'i{idx}'] = [x.value for x in hypernode['nodes']].index(intermediate['output'])

            hypernode_map[args] = hypernode
            intermediate_map[f'i{idx}'] = args
            idx += 1

        deductions = []
        for args, config in hypernode_map.items():
            canonical_arguments = args.split(',')
            arguments = [canonical_index_to_graph_index(x, intermediate_map, hypernode_map) for x in canonical_arguments]
            nodes = config.get('nodes', [])
            deductions.append(
                HyperNode(
                    hypernode_type=HyperNodeTypes.Deductive,
                    nodes=nodes,
                    arguments=arguments
                )
            )

        return cls(goal=goal, premises=premises, abductions=[], deductions=deductions)

    def to_json(self) -> Dict[str, any]:
        """
        Converts a graph into it's json structure
        :return: Json structure of the graph
        """

        if (len(self.deductions) > 0 and isinstance(self.deductions[0], str)) or\
                (len(self.abductions) > 0 and isinstance(self.abductions[0], str)):
            # If we are in the canonical variation of the graph class, return the canonical version of the to_json func.
            return self.to_canonical_json()

        return {
            'goal': self.goal.to_json(),
            'premises': [x.to_json() for x in self.premises],
            'abductions': [x.to_json() for x in self.abductions],
            'deductions': [x.to_json() for x in self.deductions],
            'missing_premises': [x.to_json() for x in self.missing_premises],
            'original_premises': [x.to_json() for x in self.__original_premises__],
            'distractor_premises': [x.to_json() for x in self.distractor_premises]
        }

    def to_canonical_json(self) -> Dict[str, any]:
        """
        Turns the graph object into its canonical json format

        :return: The json of the canonical graph
        """

        def graph_idx_to_canon_idx(index: str):
            """Helper for turning a graph index into it's canonical index"""
            assert self[index] is not None, 'Invalid index'

            key, hidx, nidx = decompose_index(index)
            if key == GraphKeyTypes.DEDUCTIVE:
                base_idx = max(sum([len(x.nodes) for x in self.deductions[0:hidx]]), 0)
                return f'i{base_idx + nidx}'
            if key == GraphKeyTypes.ABDUCTIVE:
                base_idx = max(sum([len(x.nodes) for x in self.abductions[0:hidx]]), 0)
                return f'h{base_idx + nidx}'
            if key == GraphKeyTypes.PREMISE:
                return f'p{hidx}'
            if key == GraphKeyTypes.GOAL:
                return 'g'

        intermediates = []
        for deduction in self.deductions:
            canonical_nodes = deduction.to_canonical_json()
            canonical_inputs = [graph_idx_to_canon_idx(x) for x in deduction.arguments]

            for node in canonical_nodes:
                intermediate = {'output': node['output'], 'inputs': canonical_inputs}
                intermediates.append(intermediate)

        return {
            'hypothesis': self.goal.value,
            'premises': [x.value for x in self.premises],
            'intermediates': intermediates
        }

    def __len__(self) -> int:
        """Length of the graph is determined by the number of HyperNodes in it."""
        return len(self.deductions) + len(self.abductions)

    def __getitem__(self, item) -> Union[HyperNode, Node, List[Node], List[HyperNode], List[List[Node]]]:
        """
        Given an index or index parts, find the node or hypernode that matches.

        A string index takes the form of GRAPH_KEY_TYPE:MAIN_INDEX:SUB_INDEX a helper function can be used to create it
        called "compose_index".

        Index parts can be given in lue of a string index of the form graph[GRAPH_KEY_TYPE, MAIN_INDEX, SUB_INDEX].

        If the node you want does not have a main or sub index those can be left as None.

        If you only want a hypernode, do not include the subindex.

        Using graph parts allows for multi-indexing i.e. graph[GRAPH_KEY_TYPE, MAIN_INDEX_START:MAIN_INDEX_END, :]
        similar to traditional python indexing.

        :param item: The string or index parts to use for getting the Node or HyperNode
        :return: List of Nodes or HyperNodes or a specific Node or HyperNode
        """

        if isinstance(item, str):
            key, indices, sub_indices = decompose_index(item)
            return self.get_node(key, indices, sub_indices)

        key = GraphKeyTypes[item[0]] if isinstance(item[0], str) else item[0]
        indices = None
        sub_indices = None

        if len(item) > 1:
            indices = item[1]

        if len(item) > 2:
            sub_indices = item[2]

        if indices is None and sub_indices is None:
            obj = self.get_node(key)
        elif sub_indices is None:
            obj = self.get_node(key)[indices]
        else:
            obj = [x[sub_indices] for x in self.get_node(key)[indices]]

        if isinstance(obj, list) and len(obj) == 1:
            return obj[0]
        return obj

    def get_node(
            self,
            key: GraphKeyTypes,
            index: int = None,
            sub_index: int = None
    ) -> Union[Node, HyperNode, List[Node], List[HyperNode]]:
        """
        Function that fetches a specific Node or HyperNode given a set of index parts

        :param key: The GraphKeyType to index into
        :param index: The main index (Optional, Premises will not have one)
        :param sub_index: The sub index (Optional, Premises/Goal or HyperNodes will not have one)
        :return: A single HyperNode or Node, or a list of HyperNodes or Nodes.
        """

        if key == GraphKeyTypes.PREMISE:
            if index is None:
                return self.premises
            return self.premises[index]
        if key == GraphKeyTypes.GOAL:
            return self.goal
        if key == GraphKeyTypes.MISSING_PREMISE:
            if index is None:
                return self.missing_premises
            return self.missing_premises[index]
        if key == GraphKeyTypes.ABDUCTIVE:
            if index is None:
                return self.abductions
            if sub_index is not None:
                return self.abductions[index][sub_index]
            return self.abductions[index]
        if key == GraphKeyTypes.DEDUCTIVE:
            if index is None:
                return self.deductions
            if sub_index is not None:
                return self.deductions[index][sub_index]
            return self.deductions[index]

        raise Exception(f"Unknown key: {key}")

    def get_hypernode(self, index: str) -> Optional[HyperNode]:
        """
        Attempts to return the HyperNode of a Graph Index, if None exists (i.e. a Goal or Premise) None will be returned

        :param index: The Graph Index to try to get the HyperNode of.
        :return: HyperNode of the index or None
        """

        key, idx, _ = decompose_index(index)

        if Graph.is_hypernode(compose_index(key, idx)):
            return self[key, idx]
        else:
            return None

    @staticmethod
    def is_hypernode(index: str) -> bool:
        """
        Function that can determine if an index is specific to a HyperNode (i.e. an index to a HyperNodes specific Node
        will be False because that's an index to a Node not the HyperNode).

        :param index: The index to check
        :return: Boolean of whether the index points to a HyperNode
        """

        key, index, sub_index = decompose_index(index)

        if (key == GraphKeyTypes.ABDUCTIVE or key == GraphKeyTypes.DEDUCTIVE) and sub_index is None:
            return True
        else:
            return False

    def mask(self, index: str):
        """
        Given an index, remove it from the graph (mask it) and replace it with a masked index.

        WARNING: Right now this only works for PREMISES :WARNING

        :param index: The index to mask in the graph
        :return: The masked node/hypernode at the index given
        """

        new_index = compose_index(GraphKeyTypes.MISSING_PREMISE, len(self.missing_premises))
        index_hypernode = self.get_hypernode(index)
        masked_node = self[index]

        _key, hidx, nidx = decompose_index(index)

        if _key != GraphKeyTypes.PREMISE:
            raise Exception("Masking other key types is not supported yet.")

        # Update all the intermediate steps to reflect the new masked tag.
        for hypernode in [*self.deductions, *self.abductions]:
            for idx, arg in enumerate(hypernode.arguments):
                if index_comp(index, arg):
                    hypernode.arguments[idx] = new_index

                # If an argument is higher (index wise) than the one we are masking it - reduce all indices higher than
                # the current masked index by one (shifting all of them down to account for it not being there anymore)
                if index_hypernode is not None:
                    akey, ahidx, anidx = decompose_index(arg)
                    if _key == akey and ahidx == hidx and nidx < anidx:
                        hypernode[arg][idx] = compose_index(akey, ahidx, anidx-1)
                else:
                    akey, anidx, _ = decompose_index(arg)
                    if _key == akey and hidx < anidx:
                        hypernode.arguments[idx] = compose_index(akey, anidx-1)

        if index_hypernode is not None:
            index_hypernode.nodes.remove(self[index])
        else:
            # There's really only one thing you can mask that isn't a hypernode.
            self.premises.remove(self[index])
            self.missing_premises.append(masked_node)

        return masked_node

    def reduce_to_deductions(self, node_index: str) -> 'Graph':
        """
        TODO - REFACTOR

        Given an abduction, convert it to a "generated premise" and then resolve the rest of the graph so that all the
        abductions arguments are deductive nodes.  This will return a reduced graph which only
        contains a set of deductions that lead directly to the goal + 1 additional premise that is flagged as a
        generated premise (the step at the index specified by the argument).

        :param node_index: The node we want to traverse and reduce into deductions
        :return: A reduced tree with a set of deductions that lead to the goal and a new generated premise from the
            given abduction index.
        """

        new_graph = deepcopy(self)

        key, _, _ = decompose_index(node_index)

        node = new_graph[node_index]
        hypernode = self.get_hypernode(node_index)

        # If the node we want to reduce to deductions is an abductive node we have to convert it to a deduction
        if key == GraphKeyTypes.ABDUCTIVE:
            new_graph.premises.append(Node(node.value, tags=node.tags, scores=node.scores))

            nodes = [
                Node(
                    value=new_graph[hypernode.arguments[-1]].value,
                    tags=new_graph[hypernode.arguments[-1]].tags,
                    scores=new_graph[hypernode.arguments[-1]].scores
                )
            ]

            arguments = [
                *hypernode.arguments[:-1],
                f'{compose_index(GraphKeyTypes.PREMISE, len(new_graph.premises) - 1)}'
            ]

            new_hypernode = HyperNode(hypernode_type=HyperNodeTypes.Deductive, nodes=nodes, arguments=arguments)
            new_graph.deductions.append(new_hypernode)

            new_hypernode_index = compose_index(GraphKeyTypes.DEDUCTIVE, len(new_graph.deductions) - 1, 0)

            node_last_arg_key, _, _ = decompose_index(hypernode.arguments[-1])
            if node_last_arg_key == GraphKeyTypes.ABDUCTIVE:
                new_graph.__bridge_deduction_to_abduction__(
                    new_hypernode_index,
                    hypernode.arguments[-1]
                )

            new_graph.reduce_to_deductive_subgraph(new_hypernode_index)
        else:
            new_graph.reduce_to_deductive_subgraph(node_index)

        return new_graph

    def __bridge_deduction_to_abduction__(self, deduction_idx: str, abduction_idx: str):
        """
        TODO - REFACTOR

        Helper function that converts an abductive step into a deductive step and is recursive, so if an input is an
        abductive step, that step will be converted as well..

        :param deduction_idx: The index of the deductive node that the abductive node will be made into
        :param abduction_idx: The index of the abductive node to convert.
        :return: Nothing (inplace)
        """

        abductive_hypernode = self.get_hypernode(abduction_idx)

        node = Node(
            value=self[abductive_hypernode.arguments[-1]].value,
            tags=self[abductive_hypernode.arguments[-1]].tags,
            scores=self[abductive_hypernode.arguments[-1]].scores,
        )

        hypernode = HyperNode(
            hypernode_type=HyperNodeTypes.Deductive,
            nodes=[node],
            arguments=[*abductive_hypernode.arguments[0:-1], deduction_idx]
        )

        new_node_index = compose_index(GraphKeyTypes.DEDUCTIVE, len(self.deductions) - 1, 0)

        self.deductions.append(hypernode)

        node_last_arg_key, _, _ = decompose_index(abductive_hypernode.arguments[-1])
        if node_last_arg_key == GraphKeyTypes.ABDUCTIVE:
            self.__bridge_deduction_to_abduction__(new_node_index, hypernode.arguments[-1])

    def reduce_to_deductive_subgraph(self, deductive_index: str):
        """
        TODO - REFACTOR

        Makes the current graph only have steps that lead to the given deductive_index, everything else is removed.

        :param deductive_index: Index of the deduction that the graph should traverse and create a subgraph containing
            it and it's ancestors only.
        :return: (operation done in place)
        """

        # Get the subtree of deductive nodes from the step that matched the goal
        hypernode_list = self.get_deductive_hypernode_list(deductive_index)

        # Set all the intermediates of the tree to only those that lead to directly matching the goal
        self.deductions = hypernode_list

        # For each intermediate, swap hypotheses to premises.
        for iidx, hypernode in enumerate(self.deductions):
            arguments = hypernode.arguments

            for idx, arg in enumerate(arguments):
                arg_key, _, _ = decompose_index(arg)
                if arg_key == GraphKeyTypes.ABDUCTIVE:
                    new_premise = Node(value=self[arg].value, tags={'node_type': 'generated_premise'}, scores=self[arg].scores)

                    hypernode.arguments[idx] = compose_index(GraphKeyTypes.PREMISE, len(self.premises))
                    self.premises.append(new_premise)

        # Remove the hypotheses.
        self.abductions = []

    def get_deductive_hypernode_list(self, root_index: str) -> List[HyperNode]:
        """
        TODO - REFACTOR

        Given a root deductive index, find all the deductive hypernodes below the given index in order
        and return them in an array.  It will also relabel each hypernode so that the subgraph returned is exactly
        like a normal Graph object.

        :param root_index: The top level deduction hypernode you want the sugraph for
        """

        ancestor_keys = list(set(self.get_flat_ancestors(root_index, types=[HyperNodeTypes.Deductive])))

        deductive_indices = [(idx, sub_idx) for _, idx, sub_idx in [decompose_index(x) for x in ancestor_keys]]

        deductions = sorted(zip(ancestor_keys, deductive_indices), key=lambda x: x[1][0])

        node_cache: Dict[str, str] = {}
        hypernode_cache: Dict[int, int] = {}

        reduced_deduction_list: List[HyperNode] = []

        for deductive_idx, _ in deductions:
            key, hypernode_idx, node_idx = decompose_index(deductive_idx)
            if deductive_idx in node_cache:
                pass # Already in the list?
            elif hypernode_idx in hypernode_cache:
                hypernode_idx = hypernode_cache[hypernode_idx]

                node = deepcopy(self[deductive_idx])
                reduced_deduction_list[hypernode_idx].nodes.append(node)

                new_index = compose_index(
                    GraphKeyTypes.DEDUCTIVE,
                    hypernode_idx,
                    len(reduced_deduction_list[hypernode_idx].nodes)
                )

                node_cache[deductive_idx] = new_index
            else:
                node = deepcopy(self[deductive_idx])

                original_hypernode = self.get_hypernode(deductive_idx)
                arguments = [node_cache.get(arg, arg) for arg in original_hypernode.arguments]

                hypernode = HyperNode(
                    hypernode_type=HyperNodeTypes.Deductive,
                    nodes=[node],
                    arguments=arguments
                )

                node_idx = compose_index(GraphKeyTypes.DEDUCTIVE, len(reduced_deduction_list), 0)

                node_cache[deductive_idx] = node_idx
                hypernode_cache[hypernode_idx] = len(reduced_deduction_list)

                reduced_deduction_list.append(hypernode)

        return reduced_deduction_list

    def get_flat_ancestors(self, index: str, types: List[HyperNodeTypes]) -> List[str]:
        """
        TODO - REFACTOR

        Helper function that returns a flat list of ancestor indices ([d:1:10, a:3:5, g]) etc. Filtered by the type list
        (if a type is not specified, it will not be traversed nor added to the list.

        This can contain duplicates, no filtering is applied -- it's up to the callee to interpret the subgraph.

        :param index: The index we want to traverse the ancestors of
        :param types: Accepted types to traverse and add to the ancestor list.
        """

        hypernode = self.get_hypernode(index)

        if hypernode.type not in types:
            return []

        subtree = []
        for x in hypernode.arguments:
            node_key, _, _ = decompose_index(x)

            if self.get_hypernode(x) is not None:
                subtree.extend(self.get_flat_ancestors(x, types))

        return [index, *subtree]

    def get_depth(self, index: str = None) -> int:
        """
        Get the depth of the node/hypernode at the specified index (the depth is always the max depth of the arguments
        that made a hypernode + 1)

        Depth is a measure of steps until primitive facts were used to create the step, i.e. for deductions its the
        number of steps until a step that used only premises is found. For abductions it's the same except it looks for
        when a step was created using the goal and a premise.

        :param index: The index to get the depth of.
        :return: An integer representing depth, premises are 0
        """

        if index is None:
            index = compose_index(GraphKeyTypes.DEDUCTIVE, len(self.deductions) - 1)

        if self.get_hypernode(index) is None:
            return 0

        hypernode = self.get_hypernode(index)
        return self.__get_depth__(hypernode)

    def __get_depth__(self, hypernode: HyperNode):
        """
        Helper function for the get_depth method that recurses the graph until a leaf node is hit.

        :param hypernode: The Hypernode we are traversing.
        :return: Integer value of depth for the given HyperNode
        """

        arg_depths = []
        for arg in hypernode.arguments:
            arg_hypernode = self.get_hypernode(arg)
            if arg_hypernode is not None:
                arg_depths.append(self.__get_depth__(arg_hypernode))
            else:
                arg_depths.append(0)

        return 1 + max(arg_depths)

    def slice(self, index: str, depth: int = -1) -> 'Graph':
        """
        Slices a graph into a subset of steps that are some depth away from the root index.

        One way to think of this is taking all steps that were generated around a radius/depth of the given step/index.

        WARNING: This only works for Deductive Hypernodes currently :WARNING

        :param index: The index that will act as a root of the subgraph (goal)
        :param depth: The depth to allow for the subset of steps to come from
        :return: A Graph that is a strict subset of the current graph.
        """

        if self.is_hypernode(index):
            raise Exception("Only a node can be sliced, not a hypernode.")

        _key, _, _ = decompose_index(index)
        if _key != GraphKeyTypes.DEDUCTIVE:
            raise Exception("Only deductive hypernodes can be sliced at the moment")

        root = self[index]
        goal = Node(value=root.value)
        new_graph = Graph(goal=goal, premises=[], abductions=[], deductions=[])

        if self.get_hypernode(index) is not None:
            self.__slice__(new_graph, index, depth, 0)

        return new_graph

    def __slice__(self, graph: 'Graph', index: str, depth: int, current_depth: int = 0) -> str:
        """
        Helper function for the slice method that recurses the graph adding steps are they are hit.

        :param graph: The new subset graph we are creating
        :param index: The index we want to traverse in the current graph (parent graph)
        :param depth: The max depth we are allowed to traverse
        :param current_depth: The current depth of the search
        :return: The index of the step we want to add to the subgraph
        """

        node = self[index]
        hypernode = self.get_hypernode(index)

        if hypernode is None:
            # Premise
            value = node.value
            if value not in [x.value for x in graph.premises]:
                graph.premises.append(Node(value=value))
                return compose_index(GraphKeyTypes.PREMISE, len(graph.premises) - 1)
            else:
                idx = [x.value for x in graph.premises].index(value)
                return compose_index(GraphKeyTypes.PREMISE, idx)


        for idx, deductive_hypernode in enumerate(graph.deductions):
            node_idx = deductive_hypernode.index(node)

            # Node already exists in a hypernode
            if node_idx > -1:
                return compose_index(GraphKeyTypes.DEDUCTIVE, idx, node_idx)

            # The hypernode already exists, but the node has not been added to it yet.
            if hypernode.hypernode_compare(deductive_hypernode):
                deductive_hypernode.nodes.append(node)
                return compose_index(GraphKeyTypes.DEDUCTIVE, idx, len(deductive_hypernode.nodes) - 1)


        # Hypernode does not exist.
        new_args = []
        for arg in hypernode.arguments:
            value = self[arg].value
            arg_key, _, _ = decompose_index(index)

            if depth > current_depth + 1:
                # Recurse graph for more arguments until depth is hit or we bottom out the graph.
                new_args.append(self.__slice__(graph, arg, depth, current_depth+1))
            elif value not in [x.value for x in graph.premises]:
                new_args.append(compose_index(GraphKeyTypes.PREMISE, len(graph.premises)))
                graph.premises.append(Node(value=value))
            else:
                idx = [x.value for x in graph.premises].index(value)
                new_args.append(compose_index(GraphKeyTypes.PREMISE, idx))

        hypernode = HyperNode(
            hypernode_type=HyperNodeTypes.Deductive,
            nodes=[node],
            arguments=new_args
        )
        graph.deductions.append(hypernode)
        return compose_index(GraphKeyTypes.DEDUCTIVE, len(graph.deductions) - 1, 0)

