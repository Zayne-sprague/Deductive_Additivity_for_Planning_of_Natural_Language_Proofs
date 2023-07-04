import json
from pathlib import Path
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
from fastbm25 import fastbm25
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
from jsonlines import jsonlines

CONTRASTIVE_FOLDER = Path(__file__).parent
OUTPUT_FOLDER = CONTRASTIVE_FOLDER / 'output'

def rank_passages_with_bm25(
        corpus: List[str],
        queries: List[str],
        top_k: int = 20,
):
    """
    Given a list of queries and the suggested supporting passages, this function will rank those supporting passages as
    lexical overlap according to BM25 with the original query with respect to the entire corpus.

    :param corpus: List of all the possible passages
    :param queries: List of queries or questions used to derive the supporting passages
    :param passage_indices: Indices of the supporting passages in the corpus array
    :return: Returns a list of floats for each query given (one for every supporting passage index)
    """

    tokenized_corpus = [x.split() for x in corpus]
    tokenized_queries = [x.split() for x in queries]
    bm25 = BM25Okapi(tokenized_corpus)

    sets = []
    all_scores = []
    for idx, query in tqdm(enumerate(tokenized_queries), desc='measuring sims', total=len(tokenized_queries)):
        raw_scores = bm25.get_scores(query)
        scores = raw_scores.argsort()[::-1]
        raw_scores.sort()
        sorted_scores = raw_scores[::-1][0:top_k]
        top_k_indices = scores[0:top_k]
        sets.append([corpus[x] for x in top_k_indices])
        all_scores.append(sorted_scores)

    return sets, all_scores


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--corpus_file', '-c', default=str(OUTPUT_FOLDER / 'mdr_hotpot_corpus.json'))
    parser.add_argument('--output_filename', '-o', help='json file that will contain the BM25 scores for the  support sets per question.', default='../../retrieval/output/bm25_out')
    parser.add_argument('--question_annotation_file', '-qa')

    args = parser.parse_args()

    corpus_file = Path(args.corpus_file)
    question_annotation_file = Path(args.question_annotation_file) if args.question_annotation_file else None
    output_filename = args.output_filename

    corpus = list(jsonlines.open(corpus_file))
    questions = list(jsonlines.open(question_annotation_file)) if question_annotation_file else []

    questions = questions[0:100]

    queries = [x['question'] for x in questions]
    corpus = {x['title']: x['text'] for x in corpus}
    gold_passages = [[corpus[y] for y in x['sp']] for x in questions]

    corpus_keys = list(corpus.keys())
    corpus_values = [x.lower() for x in list(corpus.values())]

    sets, set_scores = rank_passages_with_bm25(corpus_values, queries)

    if output_filename:
        output = [{
            'question': q,
            'candidate_chains': [[{'text': path, 'idx': corpus_values.index(path), 'score': score}] for path, score in zip(chain, scores)],
            'gold_args': [corpus_values[idx] for idx in q_gold],
            'gold_indices': q_gold
        } for q, chain, q_gold, scores in zip([x['question'] for x in questions], sets, [[corpus_keys.index(y) for y in x['sp']] for x in questions], set_scores)]

        output_filepath = Path(f'{output_filename}.jsonl')
        output_filepath.parent.mkdir(exist_ok=True, parents=True)

        with jsonlines.open(str(output_filepath), 'w') as f:
            f.write_all(output)

    scores = []
    for idx, s in enumerate(sets):
        g_args = gold_passages[idx]
        hits = 0
        for g in g_args:
            if g.lower() in s:
                hits+=1
        scores.append(hits)

    one_recall = sum([1 if x == 1 else 0 for x in scores])
    two_recall = sum([1 if x == 2 else 0 for x in scores])

    one_recall /= len(scores)
    two_recall /= len(scores)

    print(f'One Recall: {one_recall}')
    print(f'Two Recall: {two_recall}')
