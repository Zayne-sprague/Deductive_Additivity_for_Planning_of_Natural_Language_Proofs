from multi_type_search.search.search import Search

from tqdm import tqdm
from functools import partial


def add_tqdm_hook(search: Search, *tqdm_args, description: str = "Searching the graph", **tqdm_kwargs):
    def main_start(pbar: tqdm, *args, **kwargs):
        max_steps = args[3]
        pbar.total = max_steps

    def step_end(pbar: tqdm, *args, **kwargs):
        pbar.update(1)

    def main_end(pbar: tqdm, *args, **kwargs):
        pbar.reset()

    _pbar = tqdm(*tqdm_args, desc=description, **tqdm_kwargs)

    search.register_hook('main_start', partial(main_start, _pbar))
    search.register_hook('step_end', partial(step_end, _pbar))
    search.register_hook('main_end', partial(main_end, _pbar))
