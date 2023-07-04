import argparse
import openai
from pathlib import Path

GPT_3_FOLDER: Path = Path(__file__).parent.resolve()
MORALS_DATA_FOLDER = GPT_3_FOLDER.parent / 'data/full/morals'
TRAINED_MODELS_FOLDER = GPT_3_FOLDER.parent.resolve() / 'trained_models'

PROMPTS_FOLDER = GPT_3_FOLDER / 'prompts'
OUTPUTS_FOLDER = GPT_3_FOLDER / 'outputs'
EXPORTS_FOLDER = OUTPUTS_FOLDER / 'exports'
EXPORTS_FOLDER.mkdir(exist_ok=True, parents=True)


def gpt3_common_args(parser: argparse.ArgumentParser):
    """Set up all the common arguments shared across gpt calls."""

    default_engine = 'davinci'

    default_top_p = 0.9  # (what is this)

    default_api_key_file_path = './api_key.txt'

    parser.add_argument('--engine', '-e', type=str, default=default_engine,
                        help=f'The openai engine you want to use, default is {default_engine}.')

    parser.add_argument('--top_p', '-tp', type=float, default=default_top_p,
                        help=f'Top P for GPT-3, default is {default_top_p}')

    parser.add_argument('--api_key', type=str, required=False, help='raw string value of your gpt3 api key.')
    parser.add_argument('--api_key_file_path', type=str, required=False, default=default_api_key_file_path,
                        help=f'Path to a txt file with the GPT3 api key. Default is {default_api_key_file_path}')

    parser.add_argument('--silent', '-s', action='store_true', dest='silent',
                        help='print nothing to the console.')


def gpt3_completion_args(parser: argparse.ArgumentParser):
    default_max_output_tokens = 64

    parser.add_argument('--prompt', '-p', type=str,
                        help='Prompt to pass into the completion endpoint.')
    parser.add_argument('--prompt_file_name', '-pfn', type=str,
                        help='Name of the file in the prompts directory.')
    parser.add_argument('--num_completions', '-nc', type=int, default=1,
                        help="Number of completions to return from GPT3")
    parser.add_argument('--log_probs', '-lp', type=int, default=0,
                        help='Return the log probabilities from open ai call, you got probs-- try some logs. :)')
    parser.add_argument('--max_output_tokens', '-mot', type=int, default=default_max_output_tokens,
                        help=f'Maximum number of output tokens for GPT-3, default is {default_max_output_tokens}')
    parser.add_argument('--output_file', '-of', type=str,
                        help='The output file to save the completion, if not specified it will default to the'
                            ' prompt file name if given, otherwise it will not generate an output file.')



def get_gpt3_api_key(api_key_file_path: Path) -> str:
    """Helper to read in the api key from a txt file."""
    with api_key_file_path.open('r') as f:
        return f.read().strip()


def set_gpt3_api_key(api_key: str):
    """Small helper to set the api key for openai."""
    openai.api_key = api_key
