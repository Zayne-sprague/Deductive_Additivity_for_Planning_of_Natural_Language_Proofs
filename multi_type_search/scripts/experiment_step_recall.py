from multi_type_search.search.graph import GraphKeyTypes, compose_index
from multi_type_search.search.step_selector import VectorSpaceSelector, BFSSelector, MultiLearnedCalibratorSelector
from multi_type_search.search.search_model import NodeEmbedder, CalibratorHeuristic
from multi_type_search.search.step_type import DeductiveStepType
from multi_type_search.utils.paths import ENTAILMENT_BANK_FOLDER, DATA_FOLDER

from multi_type_search.scripts.create_shallow_graphs import create_shallow_graphs

from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import torch
from copy import deepcopy


def get_examples(
        max_examples=-1,
        max_distractors=-1,
):
    random.seed(0)

    eb_data = ENTAILMENT_BANK_FOLDER / 'task_2/test.jsonl'
    # eb_data = DATA_FOLDER / 'full/babi/t15/train_t2.json'

    examples = create_shallow_graphs(
        eb_data,
        depth=1,
        min_depth=1,
        max_depth=1,
        keep_extra_premises=True,
        canonicalize=False
    )

    examples = [x for x in examples if len(x.premises) == 2]

    if max_examples > -1 and len(examples) > max_examples:
        rand_indices = random.sample(range(0, len(examples)), min(max_examples, len(examples)))
        examples = [examples[x] for x in rand_indices]

    for example in examples:
        distractors = example.distractor_premises
        if max_distractors > -1 and len(distractors) > max_distractors:
            rand_indices = random.sample(range(0, len(distractors)), min(max_distractors, len(distractors)))
            example.distractor_premises = [distractors[x] for x in rand_indices]

    return examples

def run(
        step_selector,
        examples,
        title,
        max_examples = -1,
        max_distractors = -1,
        disable_tqdm: bool = True
):
    step_type = DeductiveStepType(None)

    ranks = []
    max_new_steps = 0

    for example in tqdm(examples, desc='Running Examples', total=len(examples), disable=disable_tqdm):
        step_selector.reset()

        correct_premises = [example[x] for x in example.deductions[0].arguments]

        distractors = example.distractor_premises

        new_premises = [*example.premises, *distractors]
        random.shuffle(new_premises)

        arg1 = new_premises.index(correct_premises[0])
        arg2 = new_premises.index(correct_premises[1])
        correct_step_args = {compose_index(GraphKeyTypes.PREMISE, arg1), compose_index(GraphKeyTypes.PREMISE, arg2)}

        example.deductions = []
        example.premises = []

        new_steps = step_type.generate_step_combinations(example, new_premises)

        example.premises = new_premises

        step_selector.add_steps(new_steps, example)
        step_selector.iter_size = len(new_steps)

        queue = next(step_selector)
        queue = [set(x.arguments) for x in queue]

        if len(queue) > max_new_steps:
            max_new_steps = len(queue)

        rank = queue.index(correct_step_args)

        # list(sorted([(x.arguments, x.score) for x in new_steps], reverse=False, key=lambda x: x[1]))
        ranks.append(rank)

    mrr = sum([1/(x+1) for x in ranks]) / max(1, len(ranks))
    print(f'{title}: MRR = {mrr:.4f}')
    return mrr

if __name__ == "__main__":
    from multi_type_search.utils.paths import TRAINED_MODELS_FOLDER

    argparser = ArgumentParser()

    argparser.add_argument('--max_examples', '-me', type=int, help='Number of examples to run', default=-1)
    argparser.add_argument('--max_distractors', '-md', type=int, help='Max number of distractors to include', default=-1)

    args = argparser.parse_args()

    max_examples = args.max_examples
    max_distractors = args.max_distractors

    examples = get_examples(
        max_examples,
        max_distractors
    )

    DEVICE = 'cuda:1'

    RUN_BFS = False
    RUN_W2V = False
    RUN_GLOVE = False
    RUN_CUSTOM = False
    RUN_SIMCSE = True
    RUN_SCSEARCH = False



    # ---- #

    SIM_METRIC = 'cosine'
    USE_NORM = False

    if RUN_BFS:
        step_selector = BFSSelector()
        run(step_selector, deepcopy(examples), 'BFS', max_examples, max_distractors)
    #

    if RUN_CUSTOM:
        node_embedder = NodeEmbedder('custom_contrastive_model', device=DEVICE if torch.cuda.is_available() else 'cpu')
        node_embedder.model.eval()
        step_selector = VectorSpaceSelector(
            node_embedder,
            deductive_nms_threshold=0.05,
            use_norm=USE_NORM,
            similarity_metric=SIM_METRIC,
        )

        run(step_selector, deepcopy(examples), 'Custom Model', max_examples, max_distractors, disable_tqdm=False)


    #

    if RUN_W2V:
        node_embedder = NodeEmbedder('word2vec_contrastive_model', device=DEVICE if torch.cuda.is_available() else 'cpu')
        step_selector = VectorSpaceSelector(
            node_embedder,
            deductive_nms_threshold=0.05,
            use_norm=USE_NORM,
            similarity_metric=SIM_METRIC
        )

        run(step_selector, deepcopy(examples), 'Word 2 Vec', max_examples, max_distractors)
    #

    if RUN_GLOVE:
        node_embedder = NodeEmbedder('glove_contrastive_model', device=DEVICE if torch.cuda.is_available() else 'cpu')
        step_selector = VectorSpaceSelector(
            node_embedder,
            deductive_nms_threshold=0.05,
            use_norm=USE_NORM,
            similarity_metric=SIM_METRIC
        )
        run(step_selector, deepcopy(examples), 'GloVe', max_examples, max_distractors)

    if RUN_SIMCSE:
        node_embedder = NodeEmbedder('diffcse_model', device=DEVICE if torch.cuda.is_available() else 'cpu')
        step_selector = VectorSpaceSelector(
            node_embedder,
            deductive_nms_threshold=0.05,
            use_norm=USE_NORM,
            similarity_metric=SIM_METRIC
        )

        run(step_selector, deepcopy(examples), 'DIFFCSE', max_examples, max_distractors, disable_tqdm=False)
        #
    #

    if RUN_SCSEARCH:
        abd_model = CalibratorHeuristic('abductive_gc', goal_conditioned=False, device=DEVICE if torch.cuda.is_available() else 'cpu')
        ded_model = CalibratorHeuristic('forward_v3_gc', goal_conditioned=True, device=DEVICE if torch.cuda.is_available() else 'cpu')
        step_selector = MultiLearnedCalibratorSelector(abductive_heuristic_model=abd_model, deductive_heuristic_model=ded_model)

        run(step_selector, deepcopy(examples), 'Learned Heuristic', max_examples, max_distractors)
