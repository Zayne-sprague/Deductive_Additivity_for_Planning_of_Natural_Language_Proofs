from multi_type_search.search.graph import Graph, decompose_index
from multi_type_search.utils.paths import DATA_FOLDER
import jsonlines
import json

file = DATA_FOLDER / 'full/morals/shallow_moral100.json'
corpus_out_file = DATA_FOLDER / 'full/morals/mdr_corpus.jsonl'
qas_out_file = DATA_FOLDER / 'full/morals/mdr_qas.jsonl'


graphs = [Graph.from_json(x) for x in (json.load(file.open('r')) if file.name.endswith('.json') else list(jsonlines.Reader(file.open('r'))))]

corpus = []
questions = []

for idx, g in enumerate(graphs):

    if len(g.deductions) != 1:
        print("Need depth 1 trees only")
        continue

    q = g.goal.normalized_value
    a = g.goal.normalized_value
    sp = [f'q{idx}_p{decompose_index(arg)[1]}' for arg in g.deductions[0].arguments]
    corpus.extend([{'title': f'q{idx}_p{pidx}', 'text': premise.normalized_value} for pidx, premise in enumerate(g.premises)])

    if len(set(sp)) != 2:
        continue

    questions.append({
        'question': q,
        'answer': a,
        'type': 'bridge',
        'sp': sp,
        '_id': idx
    })

with jsonlines.open(qas_out_file, 'w') as q:
    q.write_all(questions)
with jsonlines.open(corpus_out_file, 'w') as c:
    c.write_all(corpus)
