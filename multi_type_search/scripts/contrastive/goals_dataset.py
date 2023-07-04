from tqdm import tqdm
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


class GoalDataset(Dataset):

    examples: List[List[Union[int, List[str], str]]]

    def __init__(
            self,
            tokenizer: Callable = None
    ):
        self.tokenizer = tokenizer

    def populate_examples(
            self,
            graphs: List[Graph],
            progress_bar: bool = False,
            overwrite_existing: bool = True,
    ):
        if overwrite_existing:
            self.examples = []

        for gidx, graph in tqdm(
                enumerate(graphs),
                desc='Populating examples',
                total=len(graphs),
                disable=not progress_bar
        ):

            for deduction in graph.deductions:

                all_args = [
                    graph[x].normalized_value for x in
                    deduction.arguments
                ]

                # TODO - revisit this... maybe this enforcement of having 2 args always is unnecessary.
                if len(all_args) != 2:
                    continue

                # premise_combos = combinations(all_args, 2)
                premise_combos = [all_args]

                for args in premise_combos:
                    for node in deduction:
                        goal = node.normalized_value

                        self.examples.append([
                            goal,
                            [x for x in args],
                        ])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if self.tokenizer:
            return self.tokenizer(self.examples[idx][0]), self.examples[idx][1], self.examples[idx][0]
        return self.examples[idx][0], self.examples[idx][1], self.examples[idx][0]

    def collate_fn(self, batch):
        if self.tokenizer:
            b = {
                'input_ids': collate_tokens([s[0]['input_ids'].view(-1) for s in batch], 0),
                'attention_mask': collate_tokens([s[0]['attention_mask'].view(-1) for s in batch], 0),
            }, [s[1] for s in batch], [s[2] for s in batch]
            return b
        return [s[0] for s in batch], [s[1] for s in batch], [s[2] for s in batch]


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
