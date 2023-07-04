from multi_type_search.search.step_type import StepType, StepTypes, StepModel, base_abductive_step_config
from multi_type_search.search.graph import Node, HyperNode, HyperNodeTypes, Graph, GraphKeyTypes
from multi_type_search.search.step_selector import Step

from typing import List, Tuple, Set, Dict


class AbductiveStepType(StepType):
    """
    Step Type that is responsible for generating abductive generations.

    An Abductive Generation attempts to generate a sentence that is required for a given conclusion to be true.  It can
    be conditioned on a subset of other information which then turns the abductive step into a "what is missing to make
    this conclusion true" type of operation.

    Another way to see this is through the mathematical notation of subtraction (although this is a very loose way to
    describe abduction)

    If you have a premise P and a goal conclusion statement G, you can frame abduction as

    Abduction(P, G) == G - P -> ~P

    where ~P is a statement that would be required for the goal statement to be true.  Or

    ~P + P -> G

    ---

    Abduction("Dogs are cool.", "Dogs are cool and fun.") -> "Dogs are fun."
    """

    search_obj_type: str = 'abductive_step_type'

    name: StepTypes = StepTypes.Abductive

    def __init__(
            self,
            step_model: StepModel,
            step_configurations: List[Tuple[GraphKeyTypes, List[GraphKeyTypes]]] = None
    ):
        step_configurations = base_abductive_step_config if step_configurations is None else step_configurations
        super().__init__(step_model, step_configurations)

    def create_steps(self, new_step_combos: List[List[str]]) -> List['Step']:
        return [Step(x, self) for x in new_step_combos]

    def build_hypernodes(
            self,
            generations: List[str],
            step: Step,
    ) -> Tuple[List[Node], List[HyperNode], List[HyperNode]]:
        nodes = [Node(value=x, tags={'step_type': 'abductive'}) for x in generations]
        hypernode = HyperNode(hypernode_type=HyperNodeTypes.Abductive, nodes=nodes, arguments=step.arguments)

        return [], [hypernode], []

    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        return self.search_obj_type, {
            'step_model': self.step_model.to_json_config(),
            'step_configurations': self.step_configurations
        }

    def to(self, device: str) -> 'AbductiveStepType':
        step_model = self.step_model.to(device)
        return AbductiveStepType(
            step_model,
            self.step_configurations
        )
