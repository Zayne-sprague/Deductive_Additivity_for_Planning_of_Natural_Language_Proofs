from argparse import ArgumentParser, Namespace
from pathlib import Path
import shutil
import json
from typing import List, Dict

import torch
import random
import numpy as np

import yaml
from jsonlines import jsonlines

from multi_type_search.utils.paths import SEARCH_CONFIGS_FOLDER, SEARCH_OUTPUT_FOLDER, ROOT_FOLDER
from multi_type_search.utils.config_handler import merge_yaml_and_namespace
from multi_type_search.search.graph import Graph
from multi_type_search.scripts.create_shallow_graphs import create_shallow_graphs
from multi_type_search.scripts.create_premise_masked_graphs import create_premise_masked_graphs
from multi_type_search.scripts.search import search
from multi_type_search.scripts.compare_nodes_to_goal import compare_nodes_to_goal_from_file
from multi_type_search.scripts.compare_nodes_to_missing_premises import compare_nodes_to_missing_premises_from_file
from multi_type_search.scripts.find_proofs import find_proofs_from_file
from multi_type_search.scripts.search_stat_report import search_stat_report


def track_progress(_trackfile: Path, _progress: Dict[str, any]):
    with _trackfile.open('w') as file:
        json.dump(_progress, file)


def get_progress(_trackfile: Path):
    if not _trackfile.exists():
        return {}

    with _trackfile.open('r') as file:
        return json.load(file)


def reset_progress(_progress: Dict[str, any], reset_to: str = None):
    ckpts = [
        'init',
        'data_created',
        'searched',
        'graph_goal_comparisons',
        'graph_missing_premise_comparisons',
        'find_proofs',
        'build_search_report',
        'recover_premise_csv',
        'find_proof_trees',
        'score_proof_trees',
        'proofs_to_csv',
        'convert_proofs_for_visualizer'
    ]

    start_idx = -1 if reset_to is None else ckpts.index(reset_to)

    for i in range(start_idx, len(ckpts)):
        _progress[ckpts[i]] = False

    return _progress


def search_experiment(
    experiment_name: str,
    config_name: str,
    max_graphs: int = -1,
    experiment_root_directory: Path = SEARCH_OUTPUT_FOLDER,
    force_output: bool = False,
    resume: bool = False,
    resume_at: str = None,
    replace_config: bool = False,
    seed: int = 1,
    device: str = None

):

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    experiment_folder = experiment_root_directory / experiment_name
    data_dir = experiment_folder / 'data'
    output_dir = experiment_folder / 'output'
    config_dir = experiment_folder / 'config'
    vis_dir = experiment_folder / 'visualizations'
    vis_data_dir = vis_dir / 'data'
    reports_dir = experiment_folder / 'reports'


    assert not experiment_folder.exists() or force_output, \
        'Please specify an empty file path for the output parameter -o OR specify the force flag -f'

    # Overwrite the existing experiment (delete everything)
    if experiment_folder.exists() and not resume:
        shutil.rmtree(str(experiment_folder))

    experiment_folder.mkdir(exist_ok=True, parents=True)
    data_dir.mkdir(exist_ok=True, parents=True)
    output_dir.mkdir(exist_ok=True, parents=True)
    vis_dir.mkdir(exist_ok=True, parents=True)
    vis_data_dir.mkdir(exist_ok=True, parents=True)
    reports_dir.mkdir(exist_ok=True, parents=True)
    config_dir.mkdir(exist_ok=True, parents=True)

    trackfile = data_dir / 'progress.json'
    progress = get_progress(trackfile)
    if resume:
        progress = reset_progress(progress, reset_to=resume_at)

    orig_config_file = SEARCH_CONFIGS_FOLDER / f'{config_name}.yaml'
    config_file = config_dir / 'config.yaml'

    data_file = data_dir / 'raw_data.json'
    shallow_graphs_file = data_dir / 'shallow_graphs.json'
    masked_graphs_files = data_dir / 'masked_graphs.json'
    dataset_file = data_dir / 'experiment_dataset.json'

    searched_file = output_dir / 'searched.json'
    comparison_scores_file = output_dir / 'scored_comparisons.json'
    proofs_file = output_dir / 'proofs.json'

    search_report_file = reports_dir / 'search_report.txt'

    if replace_config:
        shutil.copyfile(str(orig_config_file), str(config_file))

    if not progress.get('data_created'):
        shutil.copyfile(str(orig_config_file), str(config_file))

        data_args = Namespace()
        data_args = merge_yaml_and_namespace(config_file, data_args, ['data_files'])

        orig_data_file = ROOT_FOLDER / data_args.orig_data_file
        if str(orig_data_file).endswith('.jsonl'):
            data = list(jsonlines.open(str(orig_data_file), 'r'))
        else:
            data = json.load(orig_data_file.open('r'))

        with data_file.open('w') as f:
            json.dump(data, f)

        data_args = Namespace()
        data_args = merge_yaml_and_namespace(config_file, data_args, ['create_shallow_graphs'])

        create_shallow_graphs(
            input_file=data_file,
            output_file=shallow_graphs_file,
            force_output=True,
            **vars(data_args)
        )

        data_args = Namespace()
        data_args = merge_yaml_and_namespace(config_file, data_args, ['create_premise_masked_graphs'])

        create_premise_masked_graphs(
            input_file=shallow_graphs_file,
            output_file=masked_graphs_files,
            force_output=True,
            **vars(data_args)
        )

        json_graphs = json.load(masked_graphs_files.open('r'))
        graphs = [Graph.from_json(t) for t in json_graphs]
        if max_graphs > -1:
            rand_indices = random.sample(range(0, len(graphs)), min(max_graphs, len(graphs)))
            graphs = [graphs[x] for x in rand_indices]

        with dataset_file.open('w') as file:
            json.dump([x.to_json() for x in graphs], file)

        progress['data_created'] = True
        track_progress(trackfile, progress)

    if not progress.get('searched'):

        search_args = Namespace()
        search_args = merge_yaml_and_namespace(config_file, search_args, ['search'])

        step_selector = search_args.step_selector
        step_types = search_args.step_type
        step_validators = getattr(search_args, 'step_validator', None)

        generation_validators = getattr(search_args, 'generation_validator', None)
        termination_criteria = getattr(search_args, 'termination_criteria', None)

        premise_retriever = getattr(search_args, 'premise_retriever', None)

        max_steps = search_args.max_steps
        torch_devices = search_args.torch_devices
        all_one_premise = search_args.all_one_premise
        mix_distractors = search_args.mix_distractors
        shuffle_premises = search_args.shuffle_premises

        search(
            dataset_file,
            searched_file,
            force_output=True,
            step_selector=step_selector,
            step_types=step_types,
            step_validators=step_validators,
            generation_validators=generation_validators,
            termination_criteria=termination_criteria,
            premise_retriever=premise_retriever,
            max_steps=max_steps,
            torch_devices=[device] if device is not None else torch_devices,
            all_one_premise=all_one_premise,
            shuffle_premises=shuffle_premises,
            mix_distractors=mix_distractors,
        )

        progress['searched'] = True
        track_progress(trackfile, progress)

    if not progress.get('graph_goal_comparisons'):
        comparison_args = Namespace()
        comparison_args = merge_yaml_and_namespace(config_file, comparison_args, ['graph_goal_comparisons'])

        torch_devices = comparison_args.torch_devices
        comparison_metric = comparison_args.comparison_metric
        use_normalized_values = comparison_args.use_normalized_values
        allowed_graphkeytypes = comparison_args.allowed_graphkeytypes

        compare_nodes_to_goal_from_file(
            input_file=searched_file,
            output_file=comparison_scores_file,
            force_output=True,
            score_name='goal_score',
            comparison_metric=comparison_metric,
            allowed_graphkeytypes=allowed_graphkeytypes,
            use_normalized_values=use_normalized_values,
            torch_devices=[device] if device is not None else torch_devices,
        )

        progress['graph_goal_comparisons'] = True
        track_progress(trackfile, progress)

    if not progress.get('graph_missing_premise_comparisons'):
        comparison_args = Namespace()
        comparison_args = merge_yaml_and_namespace(config_file, comparison_args, ['graph_missing_premise_comparisons'])

        torch_devices = comparison_args.torch_devices
        comparison_metric = comparison_args.comparison_metric
        use_normalized_values = comparison_args.use_normalized_values
        allowed_graphkeytypes = comparison_args.allowed_graphkeytypes

        compare_nodes_to_missing_premises_from_file(
            input_file=comparison_scores_file,
            output_file=comparison_scores_file,
            force_output=True,
            score_name='missing_premise_score',
            comparison_metric=comparison_metric,
            allowed_graphkeytypes=allowed_graphkeytypes,
            use_normalized_values=use_normalized_values,
            torch_devices=[device] if device is not None else torch_devices,
        )

        progress['graph_missing_premise_comparisons'] = True
        track_progress(trackfile, progress)

    if not progress.get('find_proofs'):
        proof_args = Namespace()
        proof_args = merge_yaml_and_namespace(config_file, proof_args, ['find_proofs'])

        all_proofs = []
        for proof_type in proof_args.proof_types:
            score_name = proof_type['score_name']
            threshold = proof_type['threshold']
            graph_key_types = proof_type.get('graph_key_types', None)

            proofs = find_proofs_from_file(
                comparison_scores_file,
                score_name,
                threshold,
                graph_key_types,
                updated_search_file=comparison_scores_file
            )

            for pidx, proof in enumerate(proofs):
                if len(all_proofs) <= pidx:
                    all_proofs.append([])
                all_proofs[pidx].extend(proof)

        with proofs_file.open('w') as f:
            json.dump([[y.to_json() for y in x] for x in all_proofs], f)

        total_solved = 0
        for idx, p in enumerate(all_proofs):
            if len(p) > 0:
                print(f'Graph {idx} solved')
                total_solved += 1
            else:
                print(f'Graph {idx} not solved')
        print(f'total solved: {total_solved} / {len(all_proofs)}')

        progress['find_proofs'] = True
        track_progress(trackfile, progress)

    if not progress.get('build_search_report'):
        report_args = Namespace()
        report_args = merge_yaml_and_namespace(config_file, report_args, ['search_report_args'])
        track_dupe_examples = False if not hasattr(report_args, 'track_dupe_examples') else report_args.track_dupe_examples

        search_stat_report(
            experiment_path=SEARCH_OUTPUT_FOLDER / experiment_name,
            basic_stats=report_args.basic_stats,
            duplicate_gen_stats=report_args.duplicate_gen_stats,
            expansion_stats=report_args.expansion_stats,
            premise_usage_stats=report_args.premise_usage_stats,
            self_bleu_stats=report_args.self_bleu_stats,
            self_bleu_weights=report_args.self_bleu_weights,
            self_rouge_stats=report_args.self_rouge_stats,
            proof_stats=report_args.proof_stats,
            text_report=True,
            report_file=search_report_file,
            print_report=report_args.print_report,
            device=device if device is not None else report_args.device,
            track_dupe_examples=track_dupe_examples
        )

        progress['build_search_report'] = True
        track_progress(trackfile, progress)

    if not progress.get('score_proof_trees'):

        progress['score_proof_trees'] = True
        track_progress(trackfile, progress)

    if not progress.get('proofs_to_csv'):

        progress['proofs_to_csv'] = True
        track_progress(trackfile, progress)

    if not progress.get('convert_proofs_for_visualizer'):

        progress['convert_proofs_for_visualizer'] = True
        track_progress(trackfile, progress)


if __name__ == "__main__":

    argparser = ArgumentParser()

    argparser.add_argument('--experiment_name', '-en', type=str, required=True,
                           help='Name of the experiment (folder everything will be saved under in the '
                                '--experiment_root_directory.')
    argparser.add_argument('--config_name', '-cn', type=str, required=True,
                           help='Name of the config file to use for the experiment (this will only be used for the '
                                'initial run. If you are resuming, the config saved in the experiment folder will be '
                                'used.')

    argparser.add_argument('--max_graphs', '-mg', type=int, default=-1,
                           help='Maximum number of graphs to search over')
    argparser.add_argument('--experiment_root_directory', '-erd', type=str, default=str(SEARCH_OUTPUT_FOLDER),
                           help='The root directory to save the experiment and its outputs.')

    argparser.add_argument('--force_output', '-f', dest='force_output', action='store_true',
                           help='Overwrite anything currently written in the output file path.')
    argparser.add_argument('--resume', '-r', dest='resume', action='store_true',
                           help='Resume experiment')
    argparser.add_argument('--resume_at', '-ra', type=str,
                           help='If --resume is set, this will determine where to resume the exp at')
    argparser.add_argument('--replace_config', '-rc', dest='replace_config', action='store_true',
                           help='Use the latest config specified by the config_name, not the saved config in the '
                                'experiments output folder. WARNING - probably shouldn\'t use this!')
    argparser.add_argument('--seed', '-sd', type=int, default=123,
                           help='Use this to set the random seed')
    argparser.add_argument('--device_override', '-d', type=str,
                           help='Cuda Device Override')

    args = argparser.parse_args()

    _experiment_name: str = args.experiment_name
    _config_name: str = args.config_name

    _max_graphs: int = args.max_graphs
    _experiment_root_directory: Path = Path(args.experiment_root_directory)
    _force_output: bool = args.force_output
    _resume: bool = args.resume
    _resume_at: str = args.resume_at
    _replace_config: bool = args.replace_config
    _seed: int = args.seed
    _device: str = args.device_override

    search_experiment(
        experiment_name=_experiment_name,
        config_name=_config_name,
        max_graphs=_max_graphs,
        experiment_root_directory=_experiment_root_directory,
        force_output=_force_output,
        resume=_resume,
        resume_at=_resume_at,
        replace_config=_replace_config,
        seed=_seed,
        device=_device
    )
