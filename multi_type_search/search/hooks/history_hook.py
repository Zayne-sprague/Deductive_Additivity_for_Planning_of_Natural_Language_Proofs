from multi_type_search.search.search import Search

from logging import Logger
from functools import partial
from pathlib import Path
import shutil
import os
import json


def add_history_hook(search: Search, history_directory: Path, remove_existing: bool = False):
    if remove_existing and history_directory.exists():
        shutil.rmtree(str(history_directory))

    def main_start(_history_directory: Path, *args, **kwargs):
        graph = args[0]
        _history_directory.mkdir(exist_ok=True, parents=True)

        if (_history_directory / f'{graph.primitive_name}.jsonl').exists():
            os.remove(str(_history_directory / f'{graph.primitive_name}.jsonl'))

    def generation_step_end(_history_directory: Path, *args, **kwargs):
        step_taken = args[2]
        graph = args[3]
        new_premises = args[5]
        new_abductions = args[6]
        new_deductions = args[7]

        file = _history_directory / f'{graph.primitive_name}.jsonl'
        with file.open('a+') as f:
            line = json.dumps({
                'step_taken': {
                    'arguments': step_taken.arguments,
                    'step_type': step_taken.type.name.value,
                    'score': step_taken.score
                },
                'new_generations': {
                    'premises': [x.to_json() for x in new_premises],
                    'abductions': [x.to_json() for x in new_abductions],
                    'deductions': [x.to_json() for x in new_deductions],
                }
            })
            f.write(line + '\n')
            f.close()

    search.register_hook('main_start', partial(main_start, history_directory))
    search.register_hook('generation_step_end', partial(generation_step_end, history_directory))
