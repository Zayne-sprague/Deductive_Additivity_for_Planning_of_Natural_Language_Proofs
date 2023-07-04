from gpt3.score_prompt import score
from gpt3.utils import gpt3_common_args, gpt3_completion_args, get_gpt3_api_key, set_gpt3_api_key
from multi_type_search.search.step_selector import StepSelector, Step
from multi_type_search.utils.paths import GPT_3_FOLDER
from multi_type_search.search.graph import Graph

import heapq
from typing import List, Tuple, Dict


class GPT3Selector(StepSelector):
    """
    This step selector generates a prompt to feed into GPT3 which returns the average log probabilities of the goal for
    the graph back.  This allows us to compare which steps are more likely to lead to the Goal according to GPT3.

    Multiple parameters can be set to control how the prompt itself is generated.

    WARNING: This may only work for Deductive Steps (we haven't tested with Abduction) :Warning
    """

    search_obj_type: str = 'GPT_step_selector'

    iter_size: int
    step_list: List[Step]

    goal: str
    base_context: str
    include_individual_context: bool

    def __init__(
            self,
            iter_size: int = 1,
            openai_key: str = None,
            base_context: str = "",
            include_individual_context: bool = True
    ):
        """
        :param iter_size: How many steps to sample per iteration (DEFAULT 1)
        :param openai_key: The OpenAI API Key that can be used to query GPT3
        :param base_context: The base context that will be prepended to every step regardless of the arguments.
        :param include_individual_context: Use the arguments of the step in the prompt.
        """

        self.iter_size = iter_size
        self.step_list = []

        self.openai_key = openai_key

        if not openai_key:
            openai_key = get_gpt3_api_key(GPT_3_FOLDER / 'api_key.txt')
        set_gpt3_api_key(openai_key)

        self.base_context = base_context
        self.include_individual_context = include_individual_context

    def next(self) -> List[Step]:
        # Return the first self.iter_size steps or the entire list of self.step_list if its smaller.
        return [heapq.heappop(self.step_list) for _ in range(min([len(self.step_list), self.iter_size]))]

    def add_steps(self, steps: List[Step], graph: Graph) -> None:

        for step in steps:
            individual_context = ""
            if self.include_individual_context:
                individual_context = " ".join([graph[x].value for x in step.arguments])

            full_context = f'{self.base_context}\n{individual_context}'

            step.score = score(prompt=self.goal, context=full_context)

            heapq.heappush(self.step_list, step)

    def __len__(self) -> int:
        return int(len(self.step_list) / self.iter_size) + (1 if len(self.step_list) % self.iter_size > 0 else 0)

    def reset(self) -> None:
        self.step_list = []

    def __to_config__(self) -> Tuple[str, Dict[str, any]]:
        return self.search_obj_type, {
            'iter_size': self.iter_size,
            'openai_key': self.openai_key,
            'base_context': self.base_context,
            'include_individual_context': self.include_individual_context
        }
