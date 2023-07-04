from multi_type_search.search.step_type import StepType, StepTypes, StepModel, base_deductive_step_config
from multi_type_search.search.graph import Node, HyperNode, HyperNodeTypes, Graph, GraphKeyTypes
from multi_type_search.search.step_selector import Step

from typing import List, Tuple, Set, Dict


class DeductiveStepType(StepType):
    """
    Step Type that is responsible for generating deductive generations.

    A Deductive generation attempts to combine multiple peices of information into one conclusion statement.  This can
    be seen (loosely) as the mathematical operation addition.

    For example, if you have two premise statements P1 and P2, you could combine them together to generate a new fact
    that is derived from the information in P1 and P2.

    Deduction(P1, P2) == P1 + P2 -> D

    Where D is some conclusion statement.

    Deduction("Dogs are cool.", "Dogs are fun.") -> "Dogs are cool and fun."
    """

    search_obj_type: str = 'deductive_step_type'

    name: StepTypes = StepTypes.Deductive

    def __init__(
            self,
            step_model: StepModel,
            step_configurations: List[Tuple[GraphKeyTypes, List[GraphKeyTypes]]] = None
    ):
        step_configurations = base_deductive_step_config if step_configurations is None else step_configurations
        super().__init__(step_model, step_configurations)

    def create_steps(self, new_step_combos: List[List[str]]) -> List['Step']:
        return [Step(x, self) for x in new_step_combos]

    def build_hypernodes(
            self,
            generations: List[str],
            step: Step,
    ) -> Tuple[List[Node], List[HyperNode], List[HyperNode]]:
        nodes = [Node(value=x, tags={'step_type': 'deductive'}) for x in generations]
        hypernode = HyperNode(hypernode_type=HyperNodeTypes.Deductive, nodes=nodes, arguments=step.arguments)

        return [], [], [hypernode]

    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        return self.search_obj_type, {
            'step_model': self.step_model.to_json_config(),
            'step_configurations': self.step_configurations
        }

    def to(self, device: str) -> 'DeductiveStepType':
        step_model = self.step_model.to(device)
        return DeductiveStepType(
            step_model,
            self.step_configurations
        )
