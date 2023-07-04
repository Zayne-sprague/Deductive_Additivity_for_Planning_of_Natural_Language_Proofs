from tqdm import tqdm
import time
from typing import List, Dict, Union, Callable
import torch
from torch import nn
from torch.utils.data import Dataset
import random
import json
from pathlib import Path
import pickle
from itertools import combinations

from multi_type_search.search.graph import Node, HyperNode, Graph


class CorpusDataset(Dataset):

    corpus: List[str]

    def __init__(
            self,
            tokenizer: Callable = None,
    ):
        self.tokenizer = tokenizer

    def populate_examples(
            self,
            graphs: List[Graph],
            progress_bar: bool = False,
            overwrite_existing: bool = True,
    ):
        if overwrite_existing:
            self.corpus = []

        for gidx, graph in tqdm(
                enumerate(graphs),
                desc='Populating examples',
                total=len(graphs),
                disable=not progress_bar
        ):
            if len(graph.deductions[0].arguments) != 2:
                continue
            self.corpus.extend([x.normalized_value for x in graph.premises])
        self.corpus = list(set(self.corpus))

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        if self.tokenizer:
            b = self.tokenizer(self.corpus[idx], max_length=300, return_tensors='pt', padding=True, truncation=True)
            b['strings'] = self.corpus[idx]
            return b
        else:
            return self.corpus[idx]
        #return self.tokenizer(self.corpus[idx]), self.corpus[idx]

    def collate_fn(self, batch):
        s = time.time()
        #b = self.tokenizer(batch), batch
        #b = self.tokenizer(batch)
        if self.tokenizer:
            b = {
                'input_ids': collate_tokens([s['input_ids'].view(-1) for s in batch], 0),
                'attention_mask': collate_tokens([s['attention_mask'].view(-1) for s in batch], 0),
            }, [s['strings'] for s in batch]
            return b
        else:
            return batch, batch
        #return batch

def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    if len(values[0].size()) > 1:
        values = [v.view(-1) for v in values]
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res
