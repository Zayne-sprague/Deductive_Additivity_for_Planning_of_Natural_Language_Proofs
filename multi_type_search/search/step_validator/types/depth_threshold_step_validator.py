from multi_type_search.search.step_validator import StepValidator
from multi_type_search.search.graph import Graph
from multi_type_search.search.step_selector import Step

from typing import List, Tuple, Union, Dict, Optional
import itertools


class DepthThresholdStepValidator(StepValidator):
    """
    This step validator ensures that a step is within some depth of the graphs primitive leaf nodes using the Graph
    objects builtin get_depth() method (check that method for more details on what depth is)
    """

    search_obj_type: str = 'depth_threshold_step_validator'

    threshold: int

    def __init__(
            self,
            *args,
            threshold: int = None,
            **kwargs
    ):
        """
        :param args:
        :param threshold: Maximum depth allowed for any step
        :param kwargs:
        """

        super().__init__(*args, **kwargs)
        self.threshold = threshold

    @staticmethod
    def check_step(graph: Graph, step: Step, threshold: int) -> Optional[Step]:
        """
        Given a step, check to see if any of its arguments exceed the allowed threshold

        :param graph: The graph where all the arguments live
        :param step: The current step we want to validate
        :param threshold: The depth allowed for a step to exist within the search
        :return: Either the step or None (none means the step was invalid)
        """

        if any([graph.get_depth(x) >= threshold for x in step.arguments]):
            return None

        return step

    def validate(
            self,
            graph: Graph,
            unvalidated_steps: List[Step] = (),
    ) -> List[Step]:
        """
        Given new hypotheses and intermediates, make sure the inputs used to create those steps have independent sets of
        ancestors up to some depth.

        :param graph: A graph with the nodes of the arguments in each step
        :param unvalidated_steps: Unvalidated steps
        :return: Validated steps
        """

        valid_steps = []

        for step in unvalidated_steps:
            valid_step = self.check_step(graph, step, self.threshold)
            if valid_step:
                valid_steps.append(valid_step)

        return valid_steps

    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        return self.search_obj_types, {
            'threshold': self.threshold
        }
