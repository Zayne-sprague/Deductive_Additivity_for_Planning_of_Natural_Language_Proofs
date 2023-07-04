from multi_type_search.search.search_object import SearchObject
from multi_type_search.search.graph import Node, HyperNode, Graph, GraphKeyTypes, compose_index, decompose_index
from multi_type_search.search.step_selector import StepSelector, Step
from multi_type_search.search.step_type import StepModel, STEP_CONFIG_INPUT_TYPE

from typing import List, Tuple, Dict
import itertools


class StepType(SearchObject):
    """
    StepTypes are responsible for holding the step model that will run inference on the step inputs
    as well as generating new steps when given a list of new premises, deductions or a new abduction.

    They also keep track of their respective names so you can look them up easily.
    """

    step_model: StepModel
    step_configurations: List[Tuple[GraphKeyTypes, List[GraphKeyTypes]]]
    name: str

    def __init__(
            self,
            step_model: StepModel,
            step_configurations:  List[Tuple[GraphKeyTypes, List[GraphKeyTypes]]] = ()
    ):
        """
        :param step_model: The step model to use for the step type (usually you would want to use a step model that is
            trained/associated with the step type, i.e. a StepModel Search Model trained on Abductive Data for the
            Abductive Step Type, but you are free to do whatever!)
        :param step_configurations: The list of step configurations that are allowed for the step type to take, these
            are rules that will be followed to generate new potential steps from the new generations.
        """

        self.step_model = step_model
        self.step_configurations = step_configurations

    def create_steps(self, new_step_combos: List[List[str]]) -> List['Step']:
        """
        This method is given a list of new step arguments, combine them with the StepType to create actual Step objects.

        :param new_step_combos: The list of new step arguments that are going to be made into actual Step objects.
        :return: List of Step Objects representing new steps derived from the new generations given to the StepType.
        """

        raise NotImplementedError("Implement this function for all step models, it just creates Step objects.")

    @staticmethod
    def __get_new_hypernode_indices__(
            main_index: int,
            graph_key_type: GraphKeyTypes,
            new_hypernode: HyperNode
    ):
        start_idx = 0
        hypernode_info = new_hypernode.tags.get('hypernode_info')
        if hypernode_info is not None:
            start_idx = hypernode_info.get('new_node_start_index', 0)
            _, main_index, _ = decompose_index(hypernode_info.get('graph_index'))

        indices = []
        for idx in range(len(new_hypernode.nodes[start_idx:])):
            indices.append(compose_index(graph_key_type, main_index, start_idx + idx))
        return indices

    @staticmethod
    def __get_existing_hypernode_indices__(
            main_index: int,
            graph_key_type: GraphKeyTypes,
            existing_hypernode: HyperNode
    ):
        end_idx = None
        hypernode_info = existing_hypernode.tags.get('hypernode_info')
        if hypernode_info is not None:
            end_idx = hypernode_info.get('new_node_start_index')

        indices = []
        for idx in range(len(existing_hypernode.nodes[:end_idx] if end_idx is not None else existing_hypernode)):
            indices.append(compose_index(graph_key_type, main_index, idx))
        return indices

    def generate_step_combinations(
            self,
            graph: Graph,
            new_premises: List[Node] = (),
            new_abductions: List[HyperNode] = (),
            new_deductions: List[HyperNode] = ()
    ) -> List['Step']:
        """
        This method will create a list of Step objects following the rules given in to the StepType model in
        step_configurations.

        :param graph: The current graph holds all the current generations (premises/abductions/deductions/goal)
        :param new_premises: Newly created premises
        :param new_abductions: Newly created abductions (HyperNodes)
        :param new_deductions: Newly created deductions (HyperNodes)
        :returns: List of Step Objects that are now possible given the new generations (premises/abductions/deductions)
        """

        new_premise_indices = [
            compose_index(GraphKeyTypes.PREMISE, len(graph.premises) + x) for x in range(len(new_premises))
        ]

        new_deduction_indices = [
            x for idx, hypernode in enumerate(new_deductions)
            for x in self.__get_new_hypernode_indices__(len(graph.deductions) + idx, GraphKeyTypes.DEDUCTIVE, hypernode)
        ]

        new_abduction_indices = [
            x for idx, hypernode in enumerate(new_abductions)
            for x in self.__get_new_hypernode_indices__(len(graph.abductions) + idx, GraphKeyTypes.ABDUCTIVE, hypernode)
        ]

        all_premise_indices = [
            compose_index(GraphKeyTypes.PREMISE, x) for x in range(len(graph.premises) + len(new_premise_indices))
        ]

        all_deduction_indices = [
            x for idx, hypernode in enumerate(graph.deductions)
            for x in self.__get_existing_hypernode_indices__(idx, GraphKeyTypes.DEDUCTIVE, hypernode)
        ]

        all_deduction_indices.extend(new_deduction_indices)

        all_abduction_indices = [
            x for idx, hypernode in enumerate(graph.abductions)
            for x in self.__get_existing_hypernode_indices__(idx, GraphKeyTypes.ABDUCTIVE, hypernode)
        ]

        all_abduction_indices.extend(new_abduction_indices)

        all_new_steps = []

        for new_input, config in self.step_configurations:

            if new_input == GraphKeyTypes.PREMISE:
                new_inputs = new_premise_indices
            elif new_input == GraphKeyTypes.DEDUCTIVE:
                new_inputs = new_deduction_indices
            elif new_input == GraphKeyTypes.ABDUCTIVE:
                new_inputs = new_abduction_indices
            else:
                raise Exception(f"Unknown input type in step config: {new_input}")

            arg_lists = []
            for arg_type in config:
                if arg_type == STEP_CONFIG_INPUT_TYPE:
                    arg_lists.append(new_inputs)
                elif arg_type == GraphKeyTypes.GOAL:
                    arg_lists.append([compose_index(GraphKeyTypes.GOAL)])
                elif arg_type == GraphKeyTypes.PREMISE:
                    arg_lists.append(all_premise_indices)
                elif arg_type == GraphKeyTypes.DEDUCTIVE:
                    arg_lists.append(all_deduction_indices)
                elif arg_type == GraphKeyTypes.ABDUCTIVE:
                    arg_lists.append(all_abduction_indices)
                else:
                    raise Exception(f'Unknown arg type in step config: {arg_type}')

            if len(arg_lists) == 0:
                return []

            allowed_steps = arg_lists[0]
            for list_idx in range(1, len(arg_lists)):
                allowed_steps = list(itertools.product(allowed_steps, arg_lists[list_idx]))

            all_new_steps.extend(allowed_steps)

        for hypernode in [*new_deductions, *new_abductions]:
            if hypernode.tags.get('hypernode_info') is not None:
                del hypernode.tags['hypernode_info']

        return self.create_steps(list(sorted(set(all_new_steps))))

    def build_hypernodes(
            self,
            generations: List[str],
            step: Step,
    ) -> Tuple[List[Node], List[HyperNode], List[HyperNode]]:
        """
        Method that takes a list of string generations (usually created by the step type's step model) and turns them
        into the correct generation type (HyperNode) for that class.  I.E. if you used the Deductive StepType to create
        a list of generations, then this method will turn all of those generations into one Deductive HyperNode.

        :param generations: The list of generations (preferably generated via the same StepType's StepModel)
        :param step: The step that created them.
        :return: A Tuple of newly created premises, newly created abductions, newly created deductions -- although all
            of these are returned, usually the StepType only populates one of these items.
        """

        raise NotImplementedError('Please implement the add_generations_to_fringe for StepModel')

    @staticmethod
    def format_stepmodel_input(
            values: List[str]
    ) -> str:
        """
        Helper function to take a step and convert it into the raw text input that a step model requires.
        """

        return " ".join(values)

    @classmethod
    def config_from_json(cls, type: str, arguments: Dict[str, any], device: str = 'cpu') -> 'StepType':
        from multi_type_search.search.step_type import AbductiveStepType, DeductiveStepType

        if arguments is None or len(arguments) == 0:
            arguments = {}

        if type == AbductiveStepType.search_obj_type:
            return AbductiveStepType(**arguments)
        if type == DeductiveStepType.search_obj_type:
            return DeductiveStepType(**arguments)

        raise Exception(f"Unknown step type type in config: {type}")

    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        raise NotImplemented("Please implement the to_config() func for multiprocessing and ease of saving configs.")

    def to_json_config(self) -> Dict[str, any]:
        _type, args = self.__to_config__()
        return {
            'constructor_type': 'step_type',
            'type': _type,
            'arguments': args
        }

    def to(self, device: str) -> 'StepType':
        return self
