from multi_type_search.search.graph import Graph
from multi_type_search.search.search_model import NodeEmbedder
from multi_type_search.utils.paths import ENTAILMENT_BANK_FOLDER
from multi_type_search.scripts.create_shallow_graphs import create_shallow_graphs

from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import torch
from typing import List
import itertools
from copy import deepcopy
import numpy as np

random.seed(0)


def run_distractors(embedder: NodeEmbedder, max_examples: int = -1, one_good_premise: bool = False):
    eb_data = ENTAILMENT_BANK_FOLDER / 'task_2/test.jsonl'

    examples = create_shallow_graphs(
        eb_data,
        depth=1,
        min_depth=1,
        max_depth=1,
        keep_extra_premises=True,
        canonicalize=False
    )

    if max_examples > -1 and len(examples) > max_examples:
        rand_indices = random.sample(range(0, len(examples)), min(max_examples, len(examples)))
        examples = [examples[x] for x in rand_indices]

    distractor_examples = []
    for ex in examples:
        distractors = ex.distractor_premises
        premises = ex.premises

        if one_good_premise:
            new_premises = list(itertools.product(distractors, premises))
        else:
            new_premises = list(itertools.product(distractors, distractors))

        for np in new_premises:
            new_ex = deepcopy(ex)
            new_ex.premises = np
            distractor_examples.append(new_ex)

    if max_examples > -1 and len(distractor_examples) > max_examples:
        rand_indices = random.sample(range(0, len(distractor_examples)), min(max_examples, len(distractor_examples)))
        distractor_examples = [distractor_examples[x] for x in rand_indices]

    pdists = average_embedding_dists(embedder, distractor_examples)
    return pdists


def run(embedder: NodeEmbedder, max_examples: int = -1) -> List[float]:
    eb_data = ENTAILMENT_BANK_FOLDER / 'task_2/test.jsonl'

    examples = create_shallow_graphs(
        eb_data,
        depth=1,
        min_depth=1,
        max_depth=1,
        keep_extra_premises=False,
        canonicalize=False
    )

    if max_examples > -1 and len(examples) > max_examples:
        rand_indices = random.sample(range(0, len(examples)), min(max_examples, len(examples)))
        examples = [examples[x] for x in rand_indices]

    pdists = average_embedding_dists(embedder, examples)
    return pdists


def average_embedding_dists(embedder: NodeEmbedder, graphs: List[Graph]) -> List[float]:
    def norm(x):
        # x = x - x.min()
        # x = x / x.max()
        # x = x * 2 - 1
        return x

    def dist(x, y):
        # return ((x - y) ** 2).mean()
        return torch.dot(x, y) / (torch.norm(x) * torch.norm(y))

    pdists = []

    for graph in tqdm(graphs, desc='Finding average embedding distances to goal', total=len(graphs)):
        premises = graph.premises
        goal = graph.goal

        p_embeddings = embedder.encode(premises)
        g_embedding = embedder.encode([goal])[0]

        premises_point = norm(p_embeddings.sum(0))

        pdist = dist(premises_point, g_embedding)

        pdists.append(pdist.detach().item())

    return pdists


def report(pdists: List[float], title: str, graph: bool = False):

    if graph:
        data = {"Goal Dists": pdists}
        fig, ax = plt.subplots()

        ax.boxplot(data.values(), notch=True, patch_artist=True, boxprops=dict(facecolor="C0"))
        ax.set_xticklabels(list(data.keys()))

        ax.set_title(f'{title}')

        ymin, ymax = 0, max_new_steps
        ax.set_ylim([ymin, ymax])

        plt.show()

        figfile = Path(f'./{title.replace(" ", "_")}.png')
        plt.savefig(str(figfile))

    avg = sum(pdists) / max(1, len(pdists))
    print(f'{title}: {avg:.10f}')


if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument('--max_examples', '-me', type=int, help='Number of examples to run', default=-1)
    # argparser.add_argument('--max_distractors', '-md', type=int, help='Max number of distractors to include', default=-1)
    argparser.add_argument('--device', '-d', type=str, help='Torch device to use', default='cpu')

    args = argparser.parse_args()

    max_examples = args.max_examples
    # max_distractors = args.max_distractors
    device = args.device

    # w2v = NodeEmbedder(model_name="word2vec_contrastive_model", device=device)
    #
    # w2v_dists = run(w2v, max_examples)
    # w2v_distractor_dists = run_distractors(w2v, max_examples, one_good_premise=False)
    # w2v_one_good_one_bad_distractor_dists = run_distractors(w2v, max_examples, one_good_premise=True)
    #
    # report(w2v_dists, "Word 2 Vec Normed Premise to Goal (D1)")
    # report(w2v_distractor_dists, "Word 2 Vec Normed Distractor Pairs to Goal (D1)")
    # report(w2v_one_good_one_bad_distractor_dists, "Word 2 Vec Normed 1 Premise + 1 Distractor to Goal (D1)")

    # diffcse = NodeEmbedder(model_name="diffcse_model", device=device)
    #
    # diffcse_dists = run(diffcse, max_examples)
    # diffcse_distractor_dists = run_distractors(diffcse, max_examples, one_good_premise=False)
    # diffcse_one_good_one_bad_distractor_dists = run_distractors(diffcse, max_examples, one_good_premise=True)
    #
    # report(diffcse_dists, "DiffCSE Normed Premise to Goal (D1)")
    # report(diffcse_distractor_dists, "DiffCSE Normed Distractor Pairs to Goal (D1)")
    # report(diffcse_one_good_one_bad_distractor_dists, "DiffCSE Normed 1 Premise + 1 Distractor to Goal (D1)")
    #
    custom = NodeEmbedder(model_name="custom_contrastive_model_20", device=device)

    custom_dists = run(custom, max_examples)
    custom_distractor_dists = run_distractors(custom, max_examples, one_good_premise=False)
    custom_one_good_one_bad_distractor_dists = run_distractors(custom, max_examples, one_good_premise=True)

    report(custom_dists, "Custom Encoder Normed Premise to Goal (D1)")
    report(custom_distractor_dists, "Custom Encoder Normed Distractor Pairs to Goal (D1)")
    report(custom_one_good_one_bad_distractor_dists, "Custom Encoder Normed 1 Premise + 1 Distractor to Goal (D1)")

    # glove = NodeEmbedder(model_name="glove_contrastive_model", device=device)
    #
    # glove_dists = run(glove, max_examples)
    # glove_distractor_dists = run_distractors(glove, max_examples, one_good_premise=False)
    # glove_one_good_one_bad_distractor_dists = run_distractors(glove, max_examples, one_good_premise=True)
    #
    # report(glove_dists, "GloVe Encoder Normed Premise to Goal (D1)")
    # report(glove_distractor_dists, "GloVe Encoder Normed Distractor Pairs to Goal (D1)")
    # report(glove_one_good_one_bad_distractor_dists, "GloVe Encoder Normed 1 Premise + 1 Distractor to Goal (D1)")
