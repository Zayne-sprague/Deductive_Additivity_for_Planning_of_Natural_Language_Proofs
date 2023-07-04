import random
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import json
from typing import List, Dict
import torch
from tqdm import tqdm
from jsonlines import jsonlines
import matplotlib.pyplot as plt
import collections

from multi_type_search.search.graph import Node, Graph, decompose_index, compose_index, GraphKeyTypes
from multi_type_search.search.search_model.types.contrastive import ContrastiveModel, contrastive_utils
from multi_type_search.utils.paths import TRAINED_MODELS_FOLDER, DATA_FOLDER
from multi_type_search.search.premise_retriever import PremiseRetriever, BM25PremiseRetriever
from multi_type_search.search.step_type import StepModel, DeductiveStepType, AbductiveStepType


def experiment_iterative_premise_retrieval(
        graphs: List[Graph],
        model: str,
        premises_file: Path = None,
        embedding_file: Path = None,
        recreate_embeddings_for_premises: bool = False,
        device: str = 'cpu',
        max_graphs: int = -1,
        bm25_first_step: bool = False,
        beam_size: int = 10,
        use_abductive_step_model: bool = False,
        use_deductions: bool = False
):
    assert premises_file and embedding_file or not premises_file, \
    'If a premise file is given, also specify an embedding file.'

    num_embeddings = len(np.load(str(DATA_FOLDER / 'full/hotpot/custom_index/index.npy')))

    if model == "BM25":
        retrieval_model = BM25PremiseRetriever()
    else:
        retrieval_model = load_contrastive_model(model, device)
        retrieval_model.eval()

    # if premises_file is not None and embedding_file is not None and model != 'BM25':
    #
    #     if not embedding_file.exists() or recreate_embeddings_for_premises:
    #         create_embeddings(retrieval_model, premises_file, embedding_file)

    graphs = [x for x in graphs if len(x.deductions[0].arguments) == 2]
    if max_graphs > -1:
        graphs = random.sample(graphs, max_graphs)

    meta_ans = list(jsonlines.open(str(DATA_FOLDER / 'full/hotpot/11_21_22/meta_fullwiki_val_1k_output.jsonl'), 'r'))
    meta_qs = list(jsonlines.open(str(DATA_FOLDER / 'full/hotpot/11_21_22/meta_fullwiki_val_qas.json'), 'r'))

    known_partials = 0

    all_metrics = []
    all_support_sets = []
    for graph in tqdm(graphs, desc='Retrieving from graphs', total=len(graphs)):

        raw_meta_top_candidates = [*[x[0] for x in [x for x in meta_ans if x['question'] == graph.goal.value][0]['candidate_chains']], *[x[1] for x in [x for x in meta_ans if x['question'] == graph.goal.value][0]['candidate_chains']]]
        meta_top_ans = [x['text'] for x in raw_meta_top_candidates]
        meta_top_titles = [x['title'] for x in raw_meta_top_candidates]

        meta_top_candidates = []
        seen_titles = []
        for t, a in zip(meta_top_titles, meta_top_ans):
            if t in seen_titles:
                continue
            meta_top_candidates.append({'title': t, 'text': t})
            seen_titles.append(t)

        meta_question = [x for x in meta_qs if x['question'] == graph.goal.value]
        meta_sp = meta_question[0]['sp']

        graph.premises = []
        gold_premise_indices = []
        for c in meta_top_candidates:
            n = Node(c['text'])
            graph.premises.append(n)
            if c['title'] in meta_sp:
                gold_premise_indices.append(len(graph.premises) - 1)

        if len(gold_premise_indices) == 0:
            gold_premise_indices = [-1, -1]
            known_partials += 2
        if len(gold_premise_indices) == 1:
            gold_premise_indices.append(-1)
            known_partials +=1

        assert len(gold_premise_indices) == 2, 'We parsed meta wrong'

        graph.deductions[0].arguments = [f'PREMISE:{x}' for x in gold_premise_indices]

        if model == 'BM25':
            support_sets = premise_retriever_retrieval_from_graph_beam_search(graph, retrieval_model, beam_size)
        else:
            if use_deductions:
                support_sets = retrieve_from_graph_beam_search_deduction(graph, retrieval_model, beam_size, bm25_first_step, False, None)
            else:
                stepmodel = None ##StepModel('t5_abductive_step', device=device)
                support_sets = retrieve_from_graph_beam_search(graph, retrieval_model, beam_size, bm25_first_step, use_abductive_step_model, stepmodel)
        metrics = rank_support_sets(graph, support_sets, num=num_embeddings)
        all_metrics.append(metrics)
        all_support_sets.append(support_sets)

    golds = [[int(x.replace('PREMISE:', '')) for x in y.deductions[0].arguments] for y in graphs]

    hard_misses = [(graphs[idx], x, m) for idx, (m, x) in enumerate(zip(all_metrics, all_support_sets)) if all([y['f1'] == 0. for y in m])]
    partial_misses = [(graphs[idx], x, m) for idx, (m, x) in enumerate(zip(all_metrics, all_support_sets)) if any([y['f1'] > 0. for y in m]) and all([y['f1'] < 1. for y in m])]
    hits = [(graphs[idx], x, m) for idx, (m, x) in enumerate(zip(all_metrics, all_support_sets)) if any([y['f1'] == 1. for y in m])]

    hstats = dict(collections.Counter([len(x[0].deductions[0].arguments) for x in hits]))
    pstats = dict(collections.Counter([len(x[0].deductions[0].arguments) for x in partial_misses]))
    mstats = dict(collections.Counter([len(x[0].deductions[0].arguments) for x in hard_misses]))

    print("== HIT STATS ==")
    print(hstats)
    print("== PARTIAL STATS ==")
    print(pstats)
    print("== MISS STATS ==")
    print(mstats)


    # partial_misses_premise_counts = [str(len(set([decompose_index(y)[1] for y in x[0].deductions[0].arguments]).intersection(set([a for z in x[1] for a in z])))) + "/" + str(len(x[0].deductions[0].arguments)) for x in partial_misses]
    #
    # one_out_of_two_partial_misses = [idx for idx, x in enumerate(partial_misses) if str(len(
    #     set([decompose_index(y)[1] for y in x[0].deductions[0].arguments]).intersection(
    #         set([a for z in x[1] for a in z])))) + "/" + str(len(x[0].deductions[0].arguments)) == "1/2"]
    #
    # a = [partial_misses[x][0].premises[
    #          list(set([decompose_index(y)[1] for y in partial_misses[x][0].deductions[0].arguments]).intersection(
    #              set([a for z in partial_misses[x][1] for a in z])))[0]].normalized_value for x in
    #      one_out_of_two_partial_misses]
    #
    # b = [partial_misses[x][0].premises[
    #          list(set([decompose_index(y)[1] for y in partial_misses[x][0].deductions[0].arguments]).difference(
    #              set([a for z in partial_misses[x][1] for a in z])))[0]].normalized_value for x in
    #      one_out_of_two_partial_misses]
    #
    # c = [[partial_misses[x][0].premises[z].normalized_value for y in partial_misses[x][1] for z in y if
    #       partial_misses[x][0].premises[z].normalized_value != a[idx]] for idx, x in
    #      enumerate(one_out_of_two_partial_misses)]
    # co = ["\t".join(o) for o in c]
    #
    # g = [partial_misses[x][0].goal.normalized_value for x in one_out_of_two_partial_misses]
    #
    # print("\n\n\n\n ==== HERE ==== \n\n\n\n")
    # [print(f'{l}\t{m}\t{n}\t{o}') for l, m, n, o in list(zip(g, a, b, co))]

    aggregated_metrics = aggregate_all_metrics(all_metrics)

    print(f"F1: {aggregated_metrics['f1']} | EM: {aggregated_metrics['em']} / {len(graphs)}")

    print(f'Known partials: {known_partials}')
    # context = list(jsonlines.open(str(DATA_FOLDER / 'full/hotpot/full_context.json'), 'r'))
    # golds = [[int(x.replace('PREMISE:', '')) for x in y.deductions[0].arguments] for y in graphs]
    # gold_sents = [[context[y] for y in x] for x in golds]
    # pred_sents = [[[context[z] for z in y] for y in x] for x in all_support_sets]

    # print('--- --- --')
    # line = f'Goal\tGold 1\tGold 2\tInit Fails\tComp Fails\tBeam Size'
    # for idx in range(len(pred_sents[0])):
    #     line+=f'\tFound Gold 1 (beam = {idx+1})\tFound Gold 2 (beam = {idx+1})'
    # print(line)
    # for x in list(zip([x.goal.normalized_value for x in graphs], gold_sents, pred_sents)):
    #     line = f'{x[0]}\t{x[1][0]}\t{x[1][1]}'
    #     init_fails = 0
    #     comp_fails = 0
    #     # for s in x[2]:
    #     #     line += f'\t{x[1][0] in s}\t{x[1][1] in s}'
    #     for s in x[2]:
    #         if s[0] not in x[1]:
    #             init_fails += 1
    #         if s[1] not in x[1]:
    #             comp_fails += 1
    #     line += f'\t{init_fails}\t{comp_fails}\t{len(x[2])}'
    #     print(line)


def aggregate_all_metrics(
        all_metrics: List[List[Dict[str, any]]]
):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    em = 0

    for metrics in all_metrics:
        top_m = metrics[0]
        for m in metrics[1:]:
            if not top_m['em'] and m['em']:
                top_m = m
            elif top_m['f1'] < m['f1']:
                top_m = m

        m = top_m

        tp += m['tp']
        tn += m['tn']
        fp += m['fp']
        fn += m['fn']

        em += 1 if m['em'] else 0

    p = tp / (tp + fp) if tp + fp > 0 else 0
    r = tp / (tp + fn) if tp + fn > 0 else 0

    f1 = 2 * (p * r) / (p + r) if p + r > 0 else 0

    return {'f1': f1, 'em': em}


def rank_support_sets(
        graph: Graph,
        support_sets: List[List[str]],
        num=None
):
    gold_set = []
    for y in graph.deductions[0].arguments:
        _, x, _ = decompose_index(y)
        gold_set.append(x)

    metrics = []

    for idx, support_set in enumerate(support_sets):
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        n = len(graph.premises) if num is None else num
        for idx in range(n):
            if idx in gold_set and idx in support_set:
                tp += 1
            if idx not in gold_set and idx in support_set:
                fn += 1
            if idx not in gold_set and idx not in support_set:
                tn += 1
            if idx in gold_set and idx not in support_set:
                fp += 1

        p = tp / (tp + fp) if tp + fp > 0 else 0
        r = tp / (tp + fn) if tp + fn > 0 else 0

        f1 = 2 * (p * r) / (p + r) if p + r > 0 else 0
        em = fp == 0 and fn == 0

        metrics.append({'support_set_idx': idx, 'f1': f1, 'em': em, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn})

    return metrics



def retrieve_from_graph(
        graph: Graph,
        model: ContrastiveModel
):
    """
    TODO - Make this not greedy
    :param graph:
    :param model:
    :return:
    """
    encodings = model.get_encodings([x.normalized_value for x in graph.premises])
    goal = model.get_encodings([graph.goal.normalized_value])[0]

    ranked_premises = contrastive_utils.cosine_similarity_metric(encodings, goal).argsort(descending=True)

    top_premise = ranked_premises[0].detach().item()

    support_set = [top_premise]

    new_goal = goal - encodings[top_premise]

    while len(support_set) < len(graph.deductions[0].arguments):
        ranks = contrastive_utils.cosine_similarity_metric(encodings, new_goal).argsort(descending=True)

        for i in range(len(ranks)):
            top_premise = ranks[i].detach().item()
            if top_premise not in support_set:
                break

        new_goal -= encodings[top_premise]
        support_set.append(top_premise)

    return [support_set]


def retrieve_from_graph_beam_search(
        graph: Graph,
        model: ContrastiveModel,
        k: int = 10,
        bm25_first_step: bool = False,
        use_abductive_step_model: bool = False,
        stepmodel = None
):
    """
    TODO - Make this not greedy
    :param graph:
    :param model:
    :param k: width of beam
    :param bm25_first_step:
    :param use_abductive_step_model:
    :return:
    """

    bm25 = BM25PremiseRetriever()

    # encodings = model.get_encodings([x.normalized_value for x in graph.premises])
    encodings = torch.tensor(np.load(str(DATA_FOLDER / 'full/hotpot/custom_index/index.npy'))).to(device='cuda:0')

    goal = model.get_encodings([graph.goal.normalized_value])[0]


    if not bm25_first_step:
        scores = contrastive_utils.cosine_similarity_metric(encodings, goal)
        top_premises = scores.argsort(descending=True).detach().tolist()[0:k]

        support_sets = [([x], scores[x].detach().item(), graph.goal.normalized_value) for x in top_premises]

        if use_abductive_step_model:
            new_goals_text = [stepmodel.sample(f'{graph.premises[x].normalized_value} {graph.goal.normalized_value}')[0] for x in top_premises]
            new_goals = model.get_encodings(new_goals_text)
        else:
            new_goals_text = [graph.goal.normalized_value for _ in top_premises]
            new_goals = [goal - encodings[x] for x in top_premises]

    else:
        all_premises = bm25.reduce(graph.premises, graph.goal, top_n=len(graph.premises), return_scores=True)
        top_premises = all_premises[0:k]

        support_sets = [([graph.premises.index(x[0])], 0., graph.goal.normalized_value) for x in top_premises]

        if use_abductive_step_model:
            new_goals_text = [stepmodel.sample(f'{graph.premises.index(x[0]).normalized_value} {graph.goal.normalized_value}')[0] for x in top_premises]
            new_goals = model.get_encodings(new_goals_text)
        else:
            new_goals_text = [graph.goal.normalized_value for _ in top_premises]
            new_goals = [goal - encodings[graph.premises.index(x[0])] for x in top_premises]



    i = 1
    while i < len(graph.deductions[0].arguments):
        candidates = []

        for new_goal, support_set, ngt in zip(new_goals, support_sets, new_goals_text):
            all_premises_scores = contrastive_utils.cosine_similarity_metric(encodings, new_goal)
            all_premises = all_premises_scores.argsort(descending=True).detach().tolist()
            top_premises = []

            for top_premise in all_premises:
                if top_premise not in support_set[0]:
                    top_premises.append(top_premise)
                if len(top_premises) == k:
                    break

            candidates.extend([([*support_set[0], x], support_set[1] + all_premises_scores[x].detach().item(), ngt) for x in top_premises])

        support_sets = list(sorted(candidates, key=lambda tup:tup[1], reverse=True))[:k]

        if use_abductive_step_model:
            new_goals_text = [stepmodel.sample(f'{graph.premises[s[0][-1]].normalized_value} {s[2]}')[0] for s in support_sets]
            new_goals = model.get_encodings(new_goals_text)
        else:
            new_goals = [goal + sum([-encodings[x] for x in s[0]]) for s in support_sets]
            new_goals_text = [graph.goal.normalized_value for _ in support_sets]

        i+=1

    return [x[0] for x in support_sets]



def retrieve_from_graph_beam_search_deduction(
        graph: Graph,
        model: ContrastiveModel,
        k: int = 10,
        bm25_first_step: bool = False,
        use_deductive_step_model: bool = False,
        stepmodel = None
):
    """
    TODO - Make this not greedy
    :param graph:
    :param model:
    :param k: width of beam
    :param bm25_first_step:
    :param use_deductive_step_model:
    :return:
    """

    def enc(string):
        if isinstance(string, str):
            string = [string]

        toks = model.tokenize(string)
        if isinstance(toks, dict):
            toks = {k: v.to('cuda:0') for k, v in toks.items()}
        else:
            toks.to('cuda:0')

        encs, _, _ = model(toks)
        return encs


    bm25 = BM25PremiseRetriever()

    # encodings = enc([x.normalized_value for x in graph.premises])
    encodings = torch.tensor(np.load(str(DATA_FOLDER / 'full/hotpot/11_21_22/meta_index.npy'))).to(device='cuda:0')
    goal = enc(graph.goal.normalized_value)



    if False:
        trajs = encodings + encodings.unsqueeze(1)
        trajs = trajs.view(-1, trajs.shape[-1])
        scores = contrastive_utils.cosine_similarity_metric(trajs, goal)
        pairs = list(zip((scores.argsort() / encodings.shape[0]).floor().int().tolist(),
                 (scores.argsort() % encodings.shape[0]).tolist()))

        beams = []
        seen_pairs = set()
        for p in pairs:
            if p[0] != p[1] and p not in seen_pairs:
                seen_pairs.add(p)
                beams.append([p, 1., graph.goal.normalized_value])

            if len(beams) >= k:
                return beams

    elif False:
        top_premises = list(range(k))
        support_sets = [([x], 0, graph.goal.normalized_value) for x in top_premises]
        new_goals_text = [graph.goal.normalized_value for _ in top_premises]
        new_goals = [goal for x in top_premises]

    elif not bm25_first_step:
        scores = contrastive_utils.cosine_similarity_metric(encodings, goal)
        top_premises = scores.argsort(descending=True).detach().tolist()[0:k]

        support_sets = [([x], scores[x].detach().item(), graph.goal.normalized_value) for x in top_premises]

        new_goals_text = [graph.goal.normalized_value for _ in top_premises]
        new_goals = [goal for x in top_premises]

    else:
        all_premises = bm25.reduce(graph.premises, graph.goal, top_n=len(graph.premises), return_scores=True)
        top_premises = all_premises[0:k]

        support_sets = [([graph.premises.index(x[0])], 0., graph.goal.normalized_value) for x in top_premises]

        new_goals_text = [graph.goal.normalized_value for _ in top_premises]
        new_goals = [goal - encodings[graph.premises.index(x[0])] for x in top_premises]



    i = 1
    while i < len(graph.deductions[0].arguments):
        candidates = []

        for new_goal, support_set, ngt in zip(new_goals, support_sets, new_goals_text):
            encs = sum([encodings[x] for x in support_set[0]])
            all_premises_scores = contrastive_utils.cosine_similarity_metric(encodings + encs, new_goal)
            all_premises = all_premises_scores.argsort(descending=True).detach().tolist()
            top_premises = []

            for top_premise in all_premises:
                if top_premise not in support_set[0]:
                    top_premises.append(top_premise)
                if len(top_premises) == k:
                    break

            candidates.extend([([*support_set[0], x], support_set[1] + all_premises_scores[x].detach().item(), ngt) for x in top_premises])

        support_sets = list(sorted(candidates, key=lambda tup:tup[1], reverse=True))[:k]


        new_goals = [goal for s in support_sets]
        new_goals_text = [graph.goal.normalized_value for _ in support_sets]

        i+=1

    return [x[0] for x in support_sets]





def premise_retriever_retrieval_from_graph(
        graph: Graph,
        premise_retriever: PremiseRetriever
):
    """
    TODO - Make this not greedy
    :param graph:
    :param premise_retriever:
    :return:
    """

    top_premises = premise_retriever.reduce(graph.premises, graph.goal, top_n=len(graph.premises))
    top_premise = top_premises[0]

    support_set = [graph.premises.index(top_premise)]

    new_goal = Node(f'{graph.goal.normalized_value} {top_premise.normalized_value}')

    while len(support_set) < len(graph.deductions[0].arguments):
        top_premises = premise_retriever.reduce(graph.premises, new_goal, top_n=len(graph.premises))

        for i in range(len(top_premises)):
            top_premise = top_premises[i]
            if top_premise not in support_set:
                break

        new_goal = Node(f'{new_goal.normalized_value} {top_premise.normalized_value}')
        support_set.append(graph.premises.index(top_premise))

    return [support_set]

def premise_retriever_retrieval_from_graph_beam_search(
        graph: Graph,
        premise_retriever: PremiseRetriever,
        k: int = 10
):
    """
    TODO - Make this not greedy
    :param graph:
    :param premise_retriever:
    :param k: width of beam
    :return:
    """

    all_premises = premise_retriever.reduce(graph.premises, graph.goal, top_n=len(graph.premises), return_scores=True)
    top_premises = all_premises[0:k]

    support_sets = [([graph.premises.index(x[0])], x[1]) for x in top_premises]

    new_goals = [Node(f'{graph.goal.normalized_value} {x[0].normalized_value}') for x in top_premises]

    i = 1
    while i < len(graph.deductions[0].arguments):
        candidates = []

        for new_goal, support_set in zip(new_goals, support_sets):
            all_premises = premise_retriever.reduce(graph.premises, new_goal, top_n=len(graph.premises), return_scores=True)
            top_premises = []

            for top_premise in all_premises:
                if graph.premises.index(top_premise[0]) not in support_set[0]:
                    top_premises.append(top_premise)
                if len(top_premises) == k:
                    break

            candidates.extend([([*support_set[0], graph.premises.index(x[0])], support_set[1] + x[1]) for x in top_premises])

        support_sets = list(sorted(candidates, key=lambda tup:tup[1], reverse=True))[:k]
        new_goals = [Node(f'{graph.goal.normalized_value} ' + ' '.join([graph[compose_index(GraphKeyTypes.PREMISE, x)].normalized_value for x in s[0]])) for s in support_sets]

        i+=1

    return [x[0] for x in support_sets]



def create_embeddings(
    model: ContrastiveModel,
    premises_file: Path,
    embedding_file: Path,
):
    BATCH_SIZE = 64
    batch = []

    with torch.no_grad():
        with tqdm(jsonlines.open(str(premises_file)), desc='Creating embeddings file for premises.') as pf:
            with jsonlines.open(str(embedding_file), mode='w') as ef:
                for idx, obj in enumerate(pf):
                    batch.append(obj)

                    if idx % BATCH_SIZE == 0:
                        embeddings = model.get_encodings([x['value'] for x in batch]).detach().tolist()
                        tags = [x['tag'] for x in batch]

                        batch = []

                        ef.write_all([{t: e} for t, e in zip(tags, embeddings)])


def load_contrastive_model(
        contrastive_model,
        device: str = 'cpu'
) -> ContrastiveModel:
    model = ContrastiveModel.load(TRAINED_MODELS_FOLDER / contrastive_model / 'best_checkpoint.pth', device)
    return model


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        '--input_file', '-g', type=str, required=True,
        help='List of graphs with a goal and gold steps (premises are ignored if premises_file is given)'
    )
    parser.add_argument('--premises_file', '-p', type=str, help='List of premises to search over')
    parser.add_argument('--embedding_file', '-e', type=str, help='Cache the embeddings from the premise file')
    parser.add_argument(
        '--recreate_embeddings_for_premises', '-r', action='store_true', dest='recreate_embeddings_for_premises',
        help='If passed, the embedding_file will be recreated using the premises_file'
    )
    parser.add_argument(
        '--model', '-m', type=str, required=True,
        help='Name of the contrastive model in {PROJECT_ROOT}/trained_models or BM25'
    )
    parser.add_argument(
        '--device', '-d', type=str, default='cpu'
    )
    parser.add_argument(
        '--max_graphs', '-mg', type=int, default=-1, help='Restrict the number of graphs'
    )
    parser.add_argument(
        '--seed', '-s', type=int, default=0
    )
    parser.add_argument(
        '--bm25_first_step', '-bf', action='store_true', dest='bm25_first_step', help='For the initial premises use '
                                                                                      'the bm25 ranker.'
    )
    parser.add_argument(
        '--beam_size', '-b', type=int, default=10, help='Beam size for the beam search algorithm'
    )
    parser.add_argument(
        '--use_abductive_step_model', '-a', action='store_true', dest='use_abductive_step_model',
        help='Use the abductive step model to calculate symbolic new goals rather than rely on embeddings'
    )
    parser.add_argument(
        '--use_deductions', action='store_true', dest='use_deductions',
        help='Use + instead - when planning (deductions vs abductions)'
    )

    args = parser.parse_args()

    _input_file = Path(args.input_file)
    _premises_file = Path(args.premises_file) if args.premises_file else None
    _embedding_file = Path(args.embedding_file) if args.embedding_file else None
    _recreate_embeddings_for_premises: bool = args.recreate_embeddings_for_premises
    _model: str = args.model
    _device: str = args.device
    _max_graphs: int = args.max_graphs
    _bm25_first_step: bool = args.bm25_first_step
    _beam_size: int = args.beam_size
    _use_abductive_step_model: bool = args.use_abductive_step_model
    _use_deductions: bool = args.use_deductions

    seed: int = args.seed

    random.seed(seed)
    np.random.seed(seed)

    experiment_iterative_premise_retrieval(
        graphs=[Graph.from_json(x) for x in json.load(_input_file.open('r'))],
        model=_model,
        premises_file=_premises_file,
        embedding_file=_embedding_file,
        recreate_embeddings_for_premises=_recreate_embeddings_for_premises,
        device=_device,
        max_graphs=_max_graphs,
        bm25_first_step=_bm25_first_step,
        beam_size=_beam_size,
        use_abductive_step_model=_use_abductive_step_model,
        use_deductions=_use_deductions
    )

# 3675
# 9980

# 3121
# 4990
# PR - .625
# P-EM - .111

""" beam size 10
== HIT STATS ==
{2: 554}
== PARTIAL STATS ==
{2: 2567}
== MISS STATS ==
{2: 1869}
F1: 0.3682364729458918 | EM: 554 / 4990
"""