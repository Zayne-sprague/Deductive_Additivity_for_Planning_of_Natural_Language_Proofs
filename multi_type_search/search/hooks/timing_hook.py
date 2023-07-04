from multi_type_search.search.search import Search

from pathlib import Path
from timeit import default_timer as timer
import json
from functools import partial
import statistics


class TimerTracker:

    def __init__(self, timing_directory):
        self.step_timer = None
        self.search_timer = None
        self.generation_timer = None
        self.step_times = []
        self.step_queue_population = []

        self.generation_times = []
        self.step_planning_times = []

        self.cumulative_step_times = []

        self.timing_directory = timing_directory

    def start_step(self, *args, **kwargs):
        self.step_timer = timer()

    def end_step(self, *args, **kwargs):
        step_selector = args[4]

        self.step_queue_population.append(len(step_selector))
        self.step_times.append(timer() - self.step_timer)
        self.cumulative_step_times.append(timer() - self.search_timer)

    def start_generation(self, *args, **kwargs):
        self.generation_timer = timer()

    def end_generation(self, *args, **kwargs):
        self.generation_times.append(timer() - self.generation_timer)

    def start_step_planning(self, *args, **kwargs):
        self.planning_timer = timer()

    def end_step_planning(self, *args, **kwargs):
        self.step_planning_times.append(timer() - self.planning_timer)

    def main_start(self, *args, **kwargs):
        graph = args[0]
        self.search_timer = timer()
        self.filename = graph.primitive_name

    def main_end(self, *args, **kwargs):
        total_search_time = timer() - self.search_timer

        print(f'Average Step Time: {statistics.mean(self.step_times):.2f}s')
        print(f'Average Generation Time: {statistics.mean(self.generation_times):.2f}s')
        print(f'Average Planning Time: {statistics.mean(self.step_planning_times):.2f}s')

        with (self.timing_directory / f'{self.filename}.json').open('w') as f:
            json.dump({
                'step_times': self.step_times,
                'step_queue_populations': self.step_queue_population,
                'cumulative_step_times': self.cumulative_step_times,
                'generation_times': self.generation_times,
                'total_time': total_search_time
            }, f)


def add_timing_hook(search: Search, timing_directory: Path):

    if timing_directory.exists():
        shutil.rmtree(str(timing_directory))

    timing_directory.mkdir(exist_ok=True, parents=True)

    timetracker = TimerTracker(timing_directory)

    search.register_hook('main_start', timetracker.main_start)
    search.register_hook('main_end', timetracker.main_end)
    search.register_hook('step_start', timetracker.start_step)
    search.register_hook('step_end', timetracker.end_step)
    search.register_hook('sample_generations_start', timetracker.start_generation)
    search.register_hook('sample_generations_end', timetracker.end_generation)
    search.register_hook('add_steps_start', timetracker.start_step_planning)
    search.register_hook('add_steps_end', timetracker.end_step_planning)
