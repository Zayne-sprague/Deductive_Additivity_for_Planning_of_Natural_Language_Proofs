from multi_type_search.search.graph import GraphKeyTypes

from enum import Enum


class StepTypes(Enum):
    """The current StepTypes."""
    Abductive = 'Abductive'
    Deductive = 'Deductive'


# Helper definitions for constructing abductive and deductive step combination rules.
STEP_CONFIG_INPUT_TYPE = 'input'

a = GraphKeyTypes.ABDUCTIVE
d = GraphKeyTypes.DEDUCTIVE
p = GraphKeyTypes.PREMISE
g = GraphKeyTypes.GOAL
i = STEP_CONFIG_INPUT_TYPE

A = StepTypes.Abductive
D = StepTypes.Deductive

# Rules for combining steps abductively
base_abductive_step_config = [
    (d, [i, a]),
    (p, [i, a]),
    (p, [i, g]),
    (d, [i, g]),
    (a, [d, i]),
    (a, [p, i]),
    (a, [i, g])
]

# Rules for combining steps deductively
base_deductive_step_config = [
    (p, [i, p]),
    (p, [p, i]),
    (p, [i, d]),
    (p, [d, i]),
    (d, [i, p]),
    (d, [i, d]),
    (d, [d, i])
]

