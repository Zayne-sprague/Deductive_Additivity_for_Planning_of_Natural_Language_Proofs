from argparse import ArgumentParser
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, default_data_collator
from datasets import Dataset, disable_progress_bar
disable_progress_bar()

from pathlib import Path
from typing import List
import json
from tqdm import tqdm

from multi_type_search.utils.paths import TRAINED_MODELS_FOLDER, DATA_FOLDER
from multi_type_search.search.graph import Graph, Node


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        '--model', '-m', type=str,
        help='Name of the Seq2Seq model to use. Should be in {PROJECT_ROOT}/trained_models/{name}'
    )

    args = parser.parse_args()

    model_folder: Path = TRAINED_MODELS_FOLDER / args.model

    model_config = AutoConfig.from_pretrained(
        model_folder
    )

    model_tokenizer = AutoTokenizer.from_pretrained(
        model_folder
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_folder
    )

    dataset: List[Graph] = [Graph.from_json(x) for x in json.load((DATA_FOLDER / 'full/hotpot/tmp_test.json').open('r'))]
    new_dataset = [Graph.from_json(x) for x in json.load(Path('./tmp.json').open('r'))]

    # for g in tqdm(dataset, desc='Building declaritive dataset', total=len(dataset)):
    #     g: Graph = g
    #     g.goal = Node(model_tokenizer.decode(
    #         model.generate(**model_tokenizer(g.goal.normalized_value, return_tensors='pt'))[0],
    #         skip_special_tokens=True))
    #     g.deductions[0].nodes[0] = g.goal

    # json.dump([x.to_json() for x in dataset], Path('./tmp.json').open('w'))

    print(new_dataset)




