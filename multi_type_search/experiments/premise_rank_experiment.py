from argparse import ArgumentParser
from typing import List, Dict
from pathlib import Path
import shutil
import json

from multi_type_search.scripts.experiment_premise_retrieval import rank_steps
from multi_type_search.search.step_type import StepType
from multi_type_search.search.step_validator import StepValidator
from multi_type_search.search.graph import Graph


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
        'ranked',
        'evaluate_rankings',
        'export_report'
    ]

    start_idx = -1 if reset_to is None else ckpts.index(reset_to)

    for i in range(start_idx, len(ckpts)):
        _progress[ckpts[i]] = False

    return _progress

def premise_rank_experiment(
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

        json_graphs = json.load(shallow_graphs_file.open('r'))
        graphs = [Graph.from_json(t) for t in json_graphs]
        if max_graphs > -1:
            rand_indices = random.sample(range(0, len(graphs)), min(max_graphs, len(graphs)))
            graphs = [graphs[x] for x in rand_indices]

        with dataset_file.open('w') as file:
            json.dump([x.to_json() for x in graphs], file)

        progress['data_created'] = True
        track_progress(trackfile, progress)

    if not progress.get('ranked'):

        ranked_args = Namespace()
        ranked_args = merge_yaml_and_namespace(config_file, ranked_args, ['rank'])

        step_selector = search_args.step_selector
        step_type = search_args.step_type

        torch_devices = search_args.torch_devices
        shuffle_premises = search_args.shuffle_premise

        step_selector = StepSelector.from_config('step_selector', step_selector, torch_device)
        step_type = StepType.from_config('step_type', step_type, torch_device)

        graphs = [Graph.from_json(x) for x in json.load(shallow_graphs_file.open('r'))]
        ranks = [rank_steps(x, step_selector=step_selector, step_type=step_type) for x in graphs]



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
    argparser.add_argument('--experiment_root_directory', '-erd', type=str, default=str(SEARCH_OUTPUT_FOLDER / 'premise_rank'),
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

    premise_rank_experiment(
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
