from multi_type_search.search.graph import Node, HyperNode, Graph
from multi_type_search.utils.experiment.logger_module import LoggerModule
from multi_type_search.search.step_selector import StepSelector
from multi_type_search.search.step_type import StepType
from multi_type_search.search.step_validator import StepValidator
from multi_type_search.search.generation_validator import GenerationValidator
from multi_type_search.search.termination_criteria import TerminationCriteria
from multi_type_search.search.premise_retriever import PremiseRetriever
from multi_type_search.search.search import Search
from multi_type_search.search.hooks import add_history_hook, add_tqdm_hook, add_logging_hook, add_timing_hook

import random
import json
import logging
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path
from argparse import ArgumentParser
from typing import List, Optional, Dict
from functools import partial
import torch
from multiprocessing import Pool
from copy import deepcopy

import json


def __process_wrapper__(kwargs):
    return __search__(**kwargs)


def search(
    input_file: Path,
    output_file: Path,
    force_output: bool,
    step_selector: Dict[str, any],
    step_types: List[Dict[str, any]],
    step_validators: List[Dict[str, any]],
    generation_validators: List[Dict[str, any]],
    termination_criteria: List[Dict[str, any]],
    premise_retriever: Dict[str, any],
    max_steps: int,
    torch_devices: List[str],
    all_one_premise: bool = False,
    mix_distractors: bool = False,
    shuffle_premises: bool = False,
    silent: bool = False
):
    """
    Wrapper function for __search__

    Specifically, this wrapper helps spread the search across multiple devices by splitting the number of trees to
    search over.

    This splitting is done via multiprocessing and putting each instance of the search on a new core. (might not work if
    you have more devices than cpu cores... but lucky you if that's the case :D )
    """

    assert input_file.is_file() and (str(input_file).endswith('.json') or str(input_file).endswith('.jsonl')), \
        'Please specify a correct path to a json file with a list of graphs to run the search over.'
    assert not output_file.exists() or force_output, \
        'Please specify an empty file path for the output parameter -o OR specify the force flag -f'

    if str(input_file).endswith('.jsonl'):
        json_graphs = list(jsonlines.open(str(input_file), 'r'))
    else:
        json_graphs = json.load(input_file.open('r'))
    all_graphs = [Graph.from_json(t) for t in json_graphs]
    device_graphs = []

    graphs_per_device = len(all_graphs) / len(torch_devices)
    for i in range(len(torch_devices)):
        start_idx = int(min(i * graphs_per_device, len(all_graphs) - 1))
        end_idx = int(min((i+1) * graphs_per_device, len(all_graphs)))

        if i == len(torch_devices) - 1:
            # If its the last device, always take all the trees up till the last index.
            end_idx = len(all_graphs)

        device_graphs.append(all_graphs[start_idx:end_idx])

    searches = []
    device_log_path = output_file.parent.parent / f'logs/search'
    device_log_path.mkdir(exist_ok=True, parents=True)
    log = LoggerModule(console=False, log_dir=device_log_path)

    for idx, (torch_device, graphs) in enumerate(zip(torch_devices, device_graphs)):
        log_name = str(torch_device).replace(":", "_")
        device_logger = log.build_logger(log_name, log_dir=device_log_path/log_name, remove_existing=True)

        search_args = {
            'graphs': graphs,
            'step_selector': step_selector,
            'step_types': step_types,
            'step_validators': step_validators,
            'generation_validators': generation_validators,
            'termination_criteria': termination_criteria,
            'premise_retriever': premise_retriever,
            'max_steps': max_steps,
            'torch_device': torch_device,
            'all_one_premise': all_one_premise,
            'mix_distractors': mix_distractors,
            'shuffle_premises': shuffle_premises,
            'job_idx': idx,
            'log': device_logger,
            'root_history_dir': output_file.parent.parent / f'history/',
            'root_timing_dir': output_file.parent.parent / f'timings/',
        }

        searches.append(search_args)

    if len(torch_devices) == 1:
        results = [__search__(**searches[0])]
    else:
        with Pool(len(torch_devices)) as p:
            results = p.map(__process_wrapper__, searches)

    searched_graphs = []
    for result in results:
        searched_graphs.extend(result)

    # EXPORT EVALUATIONS.
    with output_file.open('w') as f:
        json.dump([x.to_json() for x in searched_graphs], f)

    if not silent:
        print("Finished Search.")


def __search__(
    graphs: List[Graph],
    step_selector: Dict[str, any],
    step_types: List[Dict[str, any]],
    step_validators: List[Dict[str, any]],
    generation_validators: List[Dict[str, any]],
    termination_criteria: List[Dict[str, any]],
    premise_retriever: Dict[str, any],
    max_steps: int,
    torch_device: str,
    all_one_premise: bool = False,
    mix_distractors: bool = False,
    shuffle_premises: bool = False,
    job_idx: int = 0,
    log: logging.Logger = None,
    root_history_dir: Path = None,
    root_timing_dir: Path = None
) -> List[Graph]:
    """
    Wrapper for performing search on a file of trees.

    :param input_file: Path to file with graphs to search over
    :param output_file: Path to the output file that the searched graphs will be stored in
    :param step_selector: the step selector to use
    :param step_types: The supported step types
    :param step_validators: The step validators to use
    :param generation_validators: The generation validators to use
    :param termination_criteria: The termination criterias for the search
    :param premise_retriever: The premise retriever for the search.
    :param max_steps: Number of maximum search steps for each example
    :param torch_device: Torch device to use for the inner search
    :param all_one_premise: Usually used for single shot models, takes a tree and combines all of it\'s premises into
        one.
    :param mix_distractors: If a tree has distractor premises, mix them in with the trees actual premises.
    :param shuffle_premises: Randomize the premises in the graph.
    :param job_idx: if the search is a part of a multiprocessed job, this is the job index.  This helps position the
        progress bars etc.
    :param log: A logger used in the search.
    :param root_history_dir: Where to store the various history object output files for the search.
    :param root_timing_dir: Where to store the timing information for the search
    :return: List of Tree objects that have expanded intermediates and hypotheses
    """

    torch.manual_seed(0)
    random.seed(0)

    if 'cuda' in torch_device and not torch.cuda.is_available():
        torch_device = 'cpu'

    # BUILD THE SEARCH OBJECT
    search = Search()

    add_tqdm_hook(search, description='Searching the graph', leave=False, position=job_idx * 2 + 1)
    add_logging_hook(search, log)
    add_history_hook(search, history_directory=root_history_dir, remove_existing=True)
    add_timing_hook(search, timing_directory=root_timing_dir)

    searched_graphs: List[Graph] = []

    step_selector = StepSelector.from_config('step_selector', step_selector, torch_device)
    step_types = [StepType.from_config('step_type', x, torch_device) for x in step_types]
    step_validators = [StepValidator.from_config('step_validator', x, torch_device) for x in step_validators or []]
    generation_validators = [GenerationValidator.from_config('generation_validator', x, torch_device) for x in generation_validators or []]
    termination_criteria = [TerminationCriteria.from_config('termination_criteria', x, torch_device) for x in termination_criteria or []]
    premise_retriever = PremiseRetriever.from_config('premise_retriever', premise_retriever, torch_device) if premise_retriever is not None else None

    # LOOP OVER GRAPHS (MAIN LOOP)
    for graph in tqdm(
            graphs,
            desc=f'Searching Graphs on device {torch_device}',
            total=len(graphs),
            position=job_idx * 2,
            leave=False,
    ):

        if mix_distractors:
            premises = [x for x in [*graph.premises, *graph.distractor_premises]]
        else:
            premises = graph.premises

        if shuffle_premises:
            random.shuffle(premises)

        if all_one_premise:
            premises = [Node(value=" ".join(x.value for x in premises))]

        original_premises = deepcopy(premises)

        # Make sure we have no deductions or abductions that could aid the search
        graph.deductions = []
        graph.abductions = []

        graph.__original_premises__ = original_premises

        graph.premises = premises

        # Perform the actual search for the tree (trying to get that missing premise)
        searched_graph = search.search(
            graph,
            step_selector,
            step_types,
            max_steps,
            step_validators,
            generation_validators,
            termination_criteria,
            premise_retriever
        )

        step_selector.reset()
        searched_graphs.append(searched_graph)

    return searched_graphs
