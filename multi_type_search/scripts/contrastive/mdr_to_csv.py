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


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--corpus_file', '-c', default=str(OUTPUT_FOLDER / 'mdr_hotpot_corpus.json'))
    parser.add_argument('--support_sets_file', '-ss', help='jsonl file that contains support sets per question.', required=True)
    parser.add_argument('--question_annotation_file', '-qa', help='MDR puts the questions and gold args in a different file, specify it here if not in the support sets file.')

    args = parser.parse_args()

    corpus_file = Path(args.corpus_file)
    support_sets_file = Path(args.support_sets_file)
    question_annotation_file = Path(args.question_annotation_file) if args.question_annotation_file else None

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

    for idx, (q, g, c) in enumerate(zip(queries, gold_passages, candidate_passages)):
        c_text = '\t'.join(c)
        print(f'{q}\t{g[0]}\t{g[1]}\t{c_text}')

        if idx == 100:
            break

