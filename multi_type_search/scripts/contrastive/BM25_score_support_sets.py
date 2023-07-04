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
        passages: List[List[str]],
        passage_indices: List[List[int]],
        output_file: Path,
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

    query_scores = []
    for idx, query in tqdm(enumerate(tokenized_queries), desc='measuring sims', total=len(tokenized_queries)):
        scores = bm25.get_batch_scores(query, passage_indices[idx])
        query_scores.append(scores)

    output_file.parent.mkdir(exist_ok=True, parents=True)
    with output_file.open('w') as f:
        json.dump({'scores': query_scores, 'queries': queries, 'passages': passages}, f)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--corpus_file', '-c', default=str(OUTPUT_FOLDER / 'mdr_hotpot_corpus.json'))
    parser.add_argument('--support_sets_file', '-ss', help='jsonl file that contains support sets per question.', required=True)
    parser.add_argument('--output_file', '-o', help='json file that will contain the BM25 scores for the  support sets per question.', required=True)
    parser.add_argument('--question_annotation_file', '-qa', help='MDR puts the questions and gold args in a different file, specify it here if not in the support sets file.')

    args = parser.parse_args()

    corpus_file = Path(args.corpus_file)
    support_sets_file = Path(args.support_sets_file)
    question_annotation_file = Path(args.question_annotation_file) if args.question_annotation_file else None
    output_file = Path(args.output_file)

    corpus = list(jsonlines.open(corpus_file))
    support_sets = list(jsonlines.open(support_sets_file))
    questions = list(jsonlines.open(question_annotation_file)) if question_annotation_file else []

    queries = [x['question'] for x in support_sets]
    candidate_passages = [[z['text'] for y in x['candidate_chains'] for z in y] for x in support_sets]
    corpus = {x['title']: x['text'] for x in corpus}

    if question_annotation_file:
        gold_passages = [[corpus[y] for y in x['sp']] for x in questions]
    else:
        gold_passages = [[y for y in x['gold_args']] for x in support_sets]

    corpus_keys = list(corpus.keys())
    corpus_values = [x.lower() for x in list(corpus.values())]

    passage_indices = [[corpus_values.index(y.lower()) for y in x] for x in candidate_passages]
    passage_keys = [[corpus_keys[y] for y in x] for x in passage_indices]

    rank_passages_with_bm25(corpus_values, queries, candidate_passages, passage_indices, output_file)