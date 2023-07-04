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
from multi_type_search.scripts.contrastive.corpus_dataset import collate_tokens
from multi_type_search.search.search_model.types.step_model import StepModel


class ContrastiveDataset(Dataset):

    examples: List[List[Union[int, List[str], str]]]



    def __init__(
            self,
            tokenizer: Callable = None,
            extra_name: str = ''
    ):
        self.tokenizer = tokenizer

    def populate_examples(
            self,
            graphs: List[Graph],
            progress_bar: bool = False,
            overwrite_existing: bool = True,
            write_cache_name: str = None,
            read_cache_name: str = None,
            write_cache_if_not_exist: bool = False,
            sampler=None
    ):
        if overwrite_existing:
            self.examples = []

        force_write_cache = None
        if read_cache_name:
            if Path(read_cache_name).exists():
                self.examples.extend(json.load(Path(read_cache_name).open('r')))
            elif write_cache_if_not_exist:
                force_write_cache = True

        if force_write_cache or read_cache_name is None or len(self.examples) == 0:
            curr_ex_id = 0

            for gidx, graph in tqdm(
                    enumerate(graphs),
                    desc='Populating examples',
                    total=len(graphs),
                    disable=not progress_bar
            ):
                graph_goal = graph.goal.normalized_value

                g_exs = []


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
                        for step_idx, node in enumerate(deduction):
                            goal = node.normalized_value

                            if sampler:
                                samples = sampler(" ".join([x for x in args]))
                                samples = list(set([*samples, goal]))

                                for s in samples:
                                    g_exs.append([
                                        gidx,
                                        [x for x in args],
                                        s,
                                        graph_goal,
                                        curr_ex_id,
                                    ])
                                curr_ex_id+=1
                            else:
                                # self.examples.append([
                                g_exs.append([
                                    gidx,
                                    [x for x in args],
                                    goal,
                                    graph_goal,
                                    curr_ex_id,
                                ])
                                curr_ex_id+=1
                if len(g_exs) > 0:
                    self.examples.append(g_exs)
            print('done')

        if write_cache_name:
            json.dump(self.examples, Path(write_cache_name).open('w'))
        elif force_write_cache and read_cache_name:
            json.dump(self.examples, Path(read_cache_name).open('w'))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        exs = self.examples[idx]

        rets = []
        for ex in exs:
            o_args = ex[1]
            o_dgoal = ex[2]
            o_ggoal = ex[3]
            if self.tokenizer:
                ex[1] = [self.tokenizer(x, max_length=300, return_tensors='pt', padding=True, truncation=True) for x in ex[1]]
                ex[2] = self.tokenizer(ex[2], max_length=300, return_tensors='pt', padding=True, truncation=True)
                ex[3] = self.tokenizer(ex[3], max_length=300, return_tensors='pt', padding=True, truncation=True)

            rets.append([*ex[0:-1], o_args, o_dgoal, o_ggoal, ex[-1]])

        r1 = [x[0] for x in rets]
        r2 = [x[1] for x in rets]
        r3 = [x[2] for x in rets]
        r4 = [x[3] for x in rets]
        r5 = [x[4] for x in rets]
        r6 = [x[5] for x in rets]
        r7 = [x[6] for x in rets]
        r8 = [x[7] for x in rets]
        return r1, r2, r3, r4, r5, r6, r7, r8

    def collate_fn(self, batch):
        # g_indices = torch.tensor([x[0] for x in batch])
        #
        # return \
        #     positive_example_matrix(batch), \
        #     g_indices, \
        #     handle_toks([y for x in batch for y in x[1]]), \
        #     handle_toks([x[2] for x in batch]), \
        #     handle_toks([x[3] for x in batch]), \
        #     [y for x in batch for y in x[4]], \
        #     [x[5] for x in batch], \
        #     [x[6] for x in batch]
        g_indices = torch.concat([torch.tensor(x[0]) for x in batch])

        if self.tokenizer:
            return \
                positive_example_matrix(batch), \
                g_indices, \
                handle_toks([z for x in batch for y in x[1] for z in y]), \
                handle_toks([z for x in batch for z in x[2] ]), \
                handle_toks([z for x in batch for z in x[3] ]), \
                [z for x in batch for y in x[4] for z in y], \
                [z for x in batch for z in x[5]], \
                [z for x in batch for z in x[6]],\
                [z for x in batch for z in x[7]]
        return \
                positive_example_matrix(batch), \
                g_indices, \
                [z for x in batch for y in x[1] for z in y], \
                [z for x in batch for z in x[2]], \
                [z for x in batch for z in x[3]], \
                [z for x in batch for y in x[4] for z in y], \
                [z for x in batch for z in x[5]], \
                [z for x in batch for z in x[6]], \
                [z for x in batch for z in x[7]]

    def save(self, path: Path):
        with path.open('w') as f:
            json.dump(self.examples, f)

    def load(self, path: Path):
        with path.open('r') as f:
            self.examples = json.load(f)

def handle_toks(toks):
    return {
            'input_ids': collate_tokens([s['input_ids'].view(-1) for s in toks], 0),
            'attention_mask': collate_tokens([s['attention_mask'].view(-1) for s in toks], 0),
        }

def positive_example_matrix(examples) -> torch.Tensor:
    g_indices = torch.concat([torch.tensor(x[-1]) for x in examples])

    return (g_indices.unsqueeze(1) == g_indices).float()
