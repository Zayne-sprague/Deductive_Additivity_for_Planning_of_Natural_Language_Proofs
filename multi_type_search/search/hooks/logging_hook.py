from multi_type_search.search.search import Search
from multi_type_search.search.graph import Graph
from multi_type_search.search.step_selector import Step

from logging import Logger
from functools import partial


def add_logging_hook(search: Search, logger: Logger):
    def step_visualization_helper(step: Step):
        args = ", ".join(step.arguments)
        _type = step.type.name
        return f'{_type}({args})'

    def main_start(log: Logger, *args, **kwargs):
        graph = args[0]
        log.info(f'Starting the search for "{graph.goal.value}"')

    def main_end(log: Logger, *args, **kwargs):
        graph = args[0]
        log.info(f"Search finished resulting in {len(graph)} hypernodes being created.")

    def generation_step_start(log: Logger, *args, **kwargs):
        log.debug(f'Step selected for generation: {step_visualization_helper(args[2])}')

    def generation_step_end(log: Logger, *args, **kwargs):
        log.debug(f'Step {step_visualization_helper(args[2])} generated {len(args[5])} Premises, {len(args[6])} Abductions, {len(args[7])} Deductions.')

    search.register_hook('main_start', partial(main_start, logger))
    search.register_hook('main_end', partial(main_end, logger))

    search.register_hook('generation_step_start', partial(generation_step_start, logger))
    search.register_hook('generation_step_end', partial(generation_step_end, logger))
