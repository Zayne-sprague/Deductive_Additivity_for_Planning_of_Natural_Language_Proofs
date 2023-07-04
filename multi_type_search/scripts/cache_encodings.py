from argparse import ArgumentParser
from pathlib import Path
from typing import List, Dict
import json
from jsonlines import jsonlines
from tqdm import tqdm
import torch
import numpy as np

from multi_type_search.utils.paths import ROOT_FOLDER, TRAINED_MODELS_FOLDER
from multi_type_search.search.search_model.types.contrastive import ContrastiveModel
from multi_type_search.search.graph import Graph


def cache_encodings(
    model: ContrastiveModel,
    graphs: List[Graph],
    emb_out: Path,
    map_out: Path
):
    model.eval()

    emb_cache = torch.tensor([])
    map_cache = {}

    enc_idx = 0
    with torch.no_grad():
        for ex in tqdm(graphs, total=len(graphs), desc='Encoding graphs'): # -> Graph
            all_enc = [x.normalized_value for x in [*ex.premises, ex.goal]]
            all_enc = [x for x in all_enc if x not in map_cache]
            batch_size = 4

            for i in range(0, len(all_enc), batch_size):
                to_enc = all_enc[i:min(i+batch_size, len(all_enc))]

                encs = model.get_encodings(to_enc)

                emb_cache = torch.cat([emb_cache, encs.detach().cpu()], dim=0)

                for s in to_enc:
                    map_cache[s] = enc_idx
                    enc_idx += 1

    emb_cache = emb_cache.numpy()
    np.save(str(emb_out), emb_cache)

    json.dump(map_cache, map_out.open('w'))


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--model', '-m', type=str, help='Contrastive model to use', required=True)
    parser.add_argument('--dataset', '-i', type=str, help='File with graphs to encode', required=True)
    parser.add_argument('--emb_out', '-eo', type=str, help='Where to store the cached embeddings', required=True)
    parser.add_argument('--map_out', '-mo', type=str, help='Where to store the str to idx to tensor map', required=True)

    args = parser.parse_args()

    model_name: Path = TRAINED_MODELS_FOLDER / args.model
    dataset: Path = ROOT_FOLDER / args.dataset
    emb_cache_file: Path = ROOT_FOLDER / Path(args.emb_out)
    map_cache_file: Path = ROOT_FOLDER / Path(args.map_out)

    emb_cache_file.parent.mkdir(exist_ok=True, parents=True)
    map_cache_file.parent.mkdir(exist_ok=True, parents=True)

    model = ContrastiveModel.load(model_name / 'best_checkpoint.pth', 'cuda')

    if str(dataset).endswith('.jsonl'):
        data = list(jsonlines.open(str(dataset), 'r'))
    else:
        data = json.load(dataset.open('r'))

    data = [Graph.from_json(x) for x in data]

    cache_encodings(model, data, emb_cache_file, map_cache_file)

