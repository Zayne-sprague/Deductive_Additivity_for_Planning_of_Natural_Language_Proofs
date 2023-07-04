from multi_type_search.search.graph import Node, Graph, GraphKeyTypes, compose_index
from multi_type_search.search.step_selector import VectorSpaceSelector, BFSSelector, MultiLearnedCalibratorSelector
from multi_type_search.search.search_model import NodeEmbedder, CalibratorHeuristic
from multi_type_search.search.step_type import DeductiveStepType

from argparse import ArgumentParser
from typing import List
from datasets import load_dataset
from tqdm import tqdm
import random
import torch
from copy import deepcopy

random.seed(0)



def build_examples(max_examples: int = -1):
    dataset = load_dataset("babi_qa", type='en', task_no='qa15')

    examples = []
    for idx, batch in enumerate(dataset['train']):
        raw_data = batch['story']
        ind_examples = get_individual_examples(raw_data)

        for prompt, ans, premises, distractors in ind_examples:
            g = Graph(f'{prompt} {ans}', premises)
            g.distractor_premises = [Node(x) for x in distractors]
            examples.append(g)

        if idx >= max_examples and max_examples > -1:
            break

    return examples


def step_select_test(step_selector, graphs: List[Graph]):

    step_type = DeductiveStepType(None)

    ranks = []
    max_new_steps = 0

    for example in tqdm(graphs, desc='Step Select Test', total=len(graphs)):
        step_selector.reset()

        new_premises = [*example.premises, *example.distractor_premises]
        random.shuffle(new_premises)

        args = [new_premises.index(x) for x in example.premises]
        correct_step_args = set([compose_index(GraphKeyTypes.PREMISE, x) for x in args])

        example.premises = []
        example.distractors = []

        new_steps = step_type.generate_step_combinations(example, new_premises)

        example.premises = new_premises

        step_selector.add_steps(new_steps, example)
        step_selector.iter_size = len(new_steps)

        queue = next(step_selector)
        queue = [set(x.arguments) for x in queue]

        if len(queue) > max_new_steps:
            max_new_steps = len(queue)

        rank = queue.index(correct_step_args)

        ranks.append(rank)

    mrr = sum([1/(x+1) for x in ranks]) / max(1, len(ranks))
    return mrr

def get_individual_examples(item):
    text = item['text']
    types = item['type']
    answers = item['answer']
    supporting_ids = item['supporting_ids']

    facts = []
    # prompts = []
    # answers = []

    examples = []

    for ty, tx, a, s in zip(types, text, answers, supporting_ids):
        if ty == 0:
            facts.append(tx)
        else:
            premises = [facts[int(x) - 1] for x in s]
            distractors = [x for x in facts if x not in premises]
            examples.append([
                tx,
                a,
                premises,
                distractors
            ])

    return examples




if __name__ == "__main__":
    max_ex = 100

    examples = build_examples(max_ex)

    SIM_METRIC = 'euclidean_distance'
    USE_NORM = True

    node_embedder = NodeEmbedder('word2vec_contrastive_model', device='cuda:0' if torch.cuda.is_available() else 'cpu')
    step_selector = VectorSpaceSelector(
        node_embedder,
        deductive_nms_threshold=0.05,
        use_norm=USE_NORM,
        similarity_metric=SIM_METRIC
    )

    w2v_score = step_select_test(step_selector, deepcopy(examples))

    print(f"W2V MRR = {w2v_score:.4f}")

    node_embedder = NodeEmbedder('glove_contrastive_model', device='cuda:0' if torch.cuda.is_available() else 'cpu')
    step_selector = VectorSpaceSelector(
        node_embedder,
        deductive_nms_threshold=0.05,
        use_norm=USE_NORM,
        similarity_metric=SIM_METRIC
    )

    glove_score = step_select_test(step_selector, deepcopy(examples))

    print(f"GloVe MRR = {glove_score:.4f}")


    SIM_METRIC = 'euclidean_distance'
    USE_NORM = False

    node_embedder = NodeEmbedder('word2vec_contrastive_model', device='cuda:0' if torch.cuda.is_available() else 'cpu')
    step_selector = VectorSpaceSelector(
        node_embedder,
        deductive_nms_threshold=0.05,
        use_norm=USE_NORM,
        similarity_metric=SIM_METRIC
    )

    w2v_score = step_select_test(step_selector, deepcopy(examples))

    print(f"W2V MRR = {w2v_score:.4f}")

    node_embedder = NodeEmbedder('glove_contrastive_model', device='cuda:0' if torch.cuda.is_available() else 'cpu')
    step_selector = VectorSpaceSelector(
        node_embedder,
        deductive_nms_threshold=0.05,
        use_norm=USE_NORM,
        similarity_metric=SIM_METRIC
    )

    glove_score = step_select_test(step_selector, deepcopy(examples))

    print(f"GloVe MRR = {glove_score:.4f}")


    SIM_METRIC = 'cosine'
    USE_NORM = True

    node_embedder = NodeEmbedder('word2vec_contrastive_model', device='cuda:0' if torch.cuda.is_available() else 'cpu')
    step_selector = VectorSpaceSelector(
        node_embedder,
        deductive_nms_threshold=0.05,
        use_norm=USE_NORM,
        similarity_metric=SIM_METRIC
    )

    w2v_score = step_select_test(step_selector, deepcopy(examples))

    print(f"W2V MRR = {w2v_score:.4f}")

    node_embedder = NodeEmbedder('glove_contrastive_model', device='cuda:0' if torch.cuda.is_available() else 'cpu')
    step_selector = VectorSpaceSelector(
        node_embedder,
        deductive_nms_threshold=0.05,
        use_norm=USE_NORM,
        similarity_metric=SIM_METRIC
    )

    glove_score = step_select_test(step_selector, deepcopy(examples))

    print(f"GloVe MRR = {glove_score:.4f}")


    SIM_METRIC = 'cosine'
    USE_NORM = False

    node_embedder = NodeEmbedder('word2vec_contrastive_model', device='cuda:0' if torch.cuda.is_available() else 'cpu')
    step_selector = VectorSpaceSelector(
        node_embedder,
        deductive_nms_threshold=0.05,
        use_norm=USE_NORM,
        similarity_metric=SIM_METRIC
    )

    w2v_score = step_select_test(step_selector, deepcopy(examples))

    print(f"W2V MRR = {w2v_score:.4f}")

    node_embedder = NodeEmbedder('glove_contrastive_model', device='cuda:0' if torch.cuda.is_available() else 'cpu')
    step_selector = VectorSpaceSelector(
        node_embedder,
        deductive_nms_threshold=0.05,
        use_norm=USE_NORM,
        similarity_metric=SIM_METRIC
    )

    glove_score = step_select_test(step_selector, deepcopy(examples))

    print(f"GloVe MRR = {glove_score:.4f}")

    # node_embedder = NodeEmbedder('custom_contrastive_model', device='cuda:0' if torch.cuda.is_available() else 'cpu')
    # step_selector = VectorSpaceSelector(node_embedder, deductive_nms_threshold=0.05)
    #
    # examples = build_examples(max_ex)
    #
    # ccm_score = step_select_test(step_selector, deepcopy(examples))
    #
    # print(f"CCM MRR = {ccm_score:.8f}")

    # abd_model = CalibratorHeuristic('abductive_gc', goal_conditioned=False,
    #                                 device='cuda:0' if torch.cuda.is_available() else 'cpu')
    # ded_model = CalibratorHeuristic('forward_v3_gc', goal_conditioned=True,
    #                                 device='cuda:0' if torch.cuda.is_available() else 'cpu')
    # step_selector = MultiLearnedCalibratorSelector(abductive_heuristic_model=abd_model,
    #                                                deductive_heuristic_model=ded_model)
    #
    # lh_score = step_select_test(step_selector, deepcopy(examples))
    #
    # print(f"SCSearch MRR = {lh_score:.8f}")
    #

