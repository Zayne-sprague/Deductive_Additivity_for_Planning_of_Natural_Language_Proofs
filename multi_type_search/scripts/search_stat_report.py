from multi_type_search.search.graph import Node, HyperNode, HyperNodeTypes, Graph, GraphKeyTypes, decompose_index, compose_index
from multi_type_search.utils.paths import SEARCH_OUTPUT_FOLDER
from multi_type_search.scripts.search_stat_basics import search_stat_basics
from multi_type_search.scripts.search_stat_duplicate_gen import search_stat_duplicate_generations
from multi_type_search.scripts.search_stat_expansions_until_proof import search_stat_expansions_until_proof
from multi_type_search.scripts.search_stat_premise_usage import search_stat_premise_usage
from multi_type_search.scripts.search_stat_self_bleu import search_stat_self_bleu
from multi_type_search.scripts.search_stat_self_rouge import search_stat_self_rouge
from multi_type_search.scripts.search_stat_proofs import search_stat_proofs

import os
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from jsonlines import jsonlines
from argparse import ArgumentParser
from copy import deepcopy
from tqdm import tqdm
import statistics
import matplotlib.pyplot as plt


def search_stat_report(
        experiment_path: Path,
        basic_stats: bool = True,
        duplicate_gen_stats: bool = True,
        expansion_stats: bool = True,
        premise_usage_stats: bool = True,
        self_bleu_stats: bool = True,
        self_bleu_weights: List[float] = (0.33, 0.33, 0.33),
        self_rouge_stats: bool = True,
        proof_stats: bool = True,
        text_report: bool = True,
        track_dupe_examples: bool = False,
        print_report: bool = False,
        report_file: Optional[Path] = None,
        device: str = 'cpu'
):
    if basic_stats:
        search_stat_basics(experiment_path)
    if duplicate_gen_stats:
        search_stat_duplicate_generations(experiment_path, device=device, track_examples=track_dupe_examples)
    if expansion_stats:
        search_stat_expansions_until_proof(experiment_path)
    if premise_usage_stats:
        search_stat_premise_usage(experiment_path)
    if self_bleu_stats:
        search_stat_self_bleu(experiment_path, weights=self_bleu_weights)
    if self_rouge_stats:
        search_stat_self_rouge(experiment_path)
    if proof_stats:
        search_stat_proofs(experiment_path)

    if text_report:
        text_report = report(experiment_path)
        if print_report:
            print(text_report)
        if report_file is not None:
            with report_file.open('w') as f:
                f.write(text_report)


def report(experiment_path: Path):

    def avg(x):
        if x is None:
            return -1
        if len(x) == 0:
            return 0
        return sum(x) / len(x)

    vis_data_file = experiment_path / 'visualizations/data'
    proof_data_file = vis_data_file / 'proofs.json'
    basic_data_file = vis_data_file / 'basic.json'
    exact_dupe_data_file = vis_data_file / 'duplicates.json'
    expansions_data_file = vis_data_file / 'expansions.json'
    premise_usage_data_file = vis_data_file / 'premise_usage.json'
    self_bleu_data_file = vis_data_file / 'self_bleu.json'
    self_rouge_data_file = vis_data_file / 'self_rouge.json'

    total_searches = -1
    total_proofs = -1
    average_proof_count = -1

    if proof_data_file.exists():
        proof_data = json.load(proof_data_file.open('r'))
        total_searches = proof_data.get('total')
        total_proofs = proof_data.get('proofs_found')
        average_proof_count = avg(proof_data.get('Proof Counts'))

    deductive_depth = -1
    abductive_depth = -1
    deductive_sample_rate = -1
    abductive_sample_rate = -1
    if basic_data_file.exists():
        basic_data = json.load(basic_data_file.open('r'))
        deductive_depth = avg(basic_data.get('Ded Depth'))
        abductive_depth = avg(basic_data.get('Abd Depth'))
        deductive_sample_rate = avg(basic_data.get('Ded Sample Rate'))
        abductive_sample_rate = avg(basic_data.get('Abd Sample Rate'))

    percentage_of_duplicates = -1
    ex_dca = -1
    ex_aca = -1
    ex_dnp = -1
    ex_anp = -1
    hm_dca = -1
    hm_aca = -1
    hm_dnp = -1
    hm_anp = -1
    if exact_dupe_data_file.exists():
        dupe_data = json.load(exact_dupe_data_file.open('r'))
        percentage_of_duplicates = dupe_data.get('percentage_of_duplicates')

        exact_data = dupe_data.get('exact_data', {})
        rehm_data = dupe_data.get('rehm_data', {})

        ex_dca = avg(exact_data.get('Ded. Copied Arg'))
        ex_aca = avg(exact_data.get('Abd. Copied Arg'))
        ex_dnp = avg(exact_data.get('Ded. Dupes'))
        ex_anp = avg(exact_data.get('Abd. Dupes'))
        hm_dca = avg(rehm_data.get('Ded. Copied Arg'))
        hm_aca = avg(rehm_data.get('Abd. Copied Arg'))
        hm_dnp = avg(rehm_data.get('Ded. Dupes'))
        hm_anp = avg(rehm_data.get('Abd. Dupes'))

    total_expansions_until_proof = -1
    steptype_expansions_until_proof = -1
    depth_number_until_proof = -1
    average_time_to_proof = -1
    if expansions_data_file.exists():
        exp_data = json.load(expansions_data_file.open('r'))
        total_expansions_until_proof = avg(exp_data.get('Total Expansion #'))
        steptype_expansions_until_proof = avg(exp_data.get('S.T. Expansion #'))
        depth_number_until_proof = avg(exp_data.get('Depth #'))
        average_time_to_proof = avg(exp_data.get("Time to Proof"))

    avg_deductive_premise_use = -1
    avg_abductive_premise_use = -1
    avg_deductive_premise_frequency = -1
    avg_abductive_premise_frequency = -1
    if premise_usage_data_file.exists():
        pusage_data = json.load(premise_usage_data_file.open('r'))

        avg_deductive_premise_use = sum([int(k) * v for k, v in pusage_data.get('deductive_premise_use_percentages').items()]) / max(sum(pusage_data.get('deductive_premise_use_percentages').values()), 1)
        avg_abductive_premise_use = sum([int(k) * v for k, v in pusage_data.get('abductive_premise_use_percentages').items()]) / max(sum(pusage_data.get('abductive_premise_use_percentages').values()), 1)
        avg_deductive_premise_frequency = sum([int(k) * v for k, v in pusage_data.get('deductive_premise_use_frequency').items()]) / max(sum(pusage_data.get('deductive_premise_use_frequency').values()), 1)
        avg_abductive_premise_frequency = sum([int(k) * v for k, v in pusage_data.get('abductive_premise_use_frequency').items()]) / max(sum(pusage_data.get('abductive_premise_use_frequency').values()), 1)

    sb_dgen = -1
    sb_agen = -1
    sb_dsgen = -1
    sb_asgen = -1
    sb_all_gen = -1
    sb_dargs = -1
    sb_aargs = -1
    sb_all_args = -1
    if self_bleu_data_file.exists():
        data = json.load(self_bleu_data_file.open('r'))
        sb_dgen = avg(data.get('D Gens'))
        sb_agen = avg(data.get('A Gens'))
        sb_dsgen = avg(data.get('D Step'))
        sb_asgen = avg(data.get('A Step'))
        sb_all_gen = avg(data.get('All Gen'))
        sb_dargs = avg(data.get('D Args'))
        sb_aargs = avg(data.get('A Args'))
        sb_all_args = avg(data.get('All Args'))

    sr_dgen = -1
    sr_agen = -1
    sr_dsgen = -1
    sr_asgen = -1
    sr_all_gen = -1
    sr_dargs = -1
    sr_aargs = -1
    sr_all_args = -1
    if self_rouge_data_file.exists():
        data = json.load(self_rouge_data_file.open('r'))
        sr_dgen = avg(data.get('D Gens'))
        sr_agen = avg(data.get('A Gens'))
        sr_dsgen = avg(data.get('D Step'))
        sr_asgen = avg(data.get('A Step'))
        sr_all_gen = avg(data.get('All Gen'))
        sr_dargs = avg(data.get('D Args'))
        sr_aargs = avg(data.get('A Args'))
        sr_all_args = avg(data.get('All Args'))


    return f"""
+============================================================+
|                        SEARCH REPORT                       |
+------------------------------------------------------------+
    Proof Coverage: {total_proofs} / {total_searches} ({total_proofs / max(total_searches, 1) * 100 :.2f}%)
    Average Proofs Per Graph: {average_proof_count:.2f}
+------------------------------------------------------------+
    Avg Deductive Depth: {deductive_depth:.2f}
    Avg Abductive Depth: {abductive_depth:.2f}
    Avg Deductive Sample Rate: {deductive_sample_rate:.2f}
    Avg Abductive Sample Rate: {abductive_sample_rate:.2f}
+------------------------------------------------------------+
    Exact Duplicates: {percentage_of_duplicates * 100:.2f}%
    
    Exact Comparisons Metrics:
        Deductive Arguments Duplicated: {ex_dca*100:.2f}%
        Abductive Arguments Duplicated: {ex_aca*100:.2f}%
        Deductive Dupe Node Percentage: {ex_dnp*100:.2f}%
        Abductive Dupe Node Percentage: {ex_anp*100:.2f}%
        
    Rouge+Entailment HM Comparisons Metrics:
        Deductive Arguments Duplicated: {hm_dca*100:.2f}%
        Abductive Arguments Duplicated: {hm_aca*100:.2f}%
        Deductive Dupe Node Percentage: {hm_dnp*100:.2f}%
        Abductive Dupe Node Percentage: {hm_anp*100:.2f}%        
+------------------------------------------------------------+
    Average Expansions Until Proof: {total_expansions_until_proof:.2f}
    Average StepType Expansions Until Proof: {steptype_expansions_until_proof:.2f}
    Average Depth Until Proof: {depth_number_until_proof:.2f}
    Average Time Until Proof: {average_time_to_proof:.2f}s
+------------------------------------------------------------+
    Avg Deductive Premise Coverage: {avg_deductive_premise_use:.2f}%
    Avg Abductive Premise Coverage: {avg_abductive_premise_use:.2f}%
    Avg Deductive Premise Frequency: {avg_deductive_premise_frequency:.2f}
    Avg Abductive Premise Frequency: {avg_abductive_premise_frequency:.2f}
+------------------------------------------------------------+
    Self-Bleu Scores:
        Deductive All Generations: {sb_dgen:.2f}
        Abductive All Generations: {sb_agen:.2f}
        Deductive Per-Step Generations: {sb_dsgen:.2f}
        Abductive Per-Step Generations: {sb_asgen:.2f}
        All Generations: {sb_all_gen:.2f}
        Deductive Arguments: {sb_dargs:.2f}
        Abductive Arguments: {sb_aargs:.2f}
        All Arguments: {sb_all_args:.2f}
    
    Self-Rouge Scores:
       Deductive All Generations: {sr_dgen:.2f}
        Abductive All Generations: {sr_agen:.2f}
        Deductive Per-Step Generations: {sr_dsgen:.2f}
        Abductive Per-Step Generations: {sr_asgen:.2f}
        All Generations: {sr_all_gen:.2f}
        Deductive Arguments: {sr_dargs:.2f}
        Abductive Arguments: {sr_aargs:.2f}
        All Arguments: {sr_all_args:.2f} 
+============================================================+
"""


if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument('--experiment_name', '-en', type=str,
                           help='Name of experiment')
    argparser.add_argument('--basic_stats', '-bs', action='store_true', dest='basic_stats',
                           help='Run basic stats')
    argparser.add_argument('--duplicate_gen_stats', '-dgs', action='store_true', dest='duplicate_gen_stats',
                           help='Run duplicate generation stats')
    argparser.add_argument('--expansion_stats', '-es', action='store_true', dest='expansion_stats',
                           help='Run expansion stats')
    argparser.add_argument('--premise_usage_stats', '-pus', action='store_true', dest='premise_usage_stats',
                           help='Run premise usage stats')
    argparser.add_argument('--self_bleu_stats', '-sbs', action='store_true', dest='self_bleu_stats',
                           help='Run self bleu stats')
    argparser.add_argument('--self_rouge_stats', '-srs', action='store_true', dest='self_rouge_stats',
                           help='Run self rouge stats')
    argparser.add_argument('--proof_stats', '-ps', action='store_true', dest='proof_stats',
                           help='Run proof stats')
    argparser.add_argument('--text_report', '-tr', action='store_true', dest='text_report',
                           help='Generate a textual report')

    args = argparser.parse_args()

    _experiment_path: Path = SEARCH_OUTPUT_FOLDER / args.experiment_name
    _basic_stats: bool = args.basic_stats
    _duplicate_gen_stats: bool = args.duplicate_gen_stats
    _expansion_stats: bool = args.expansion_stats
    _premise_usage_stats: bool = args.premise_usage_stats
    _self_bleu_stats: bool = args.self_bleu_stats
    _self_rouge_stats: bool = args.self_rouge_stats
    _proof_stats: bool = args.proof_stats
    _text_report: bool = args.text_report

    search_stat_report(
        experiment_path=_experiment_path,
        basic_stats=_basic_stats,
        duplicate_gen_stats=_duplicate_gen_stats,
        expansion_stats=_expansion_stats,
        premise_usage_stats=_premise_usage_stats,
        self_bleu_stats=_self_bleu_stats,
        self_rouge_stats=_self_rouge_stats,
        proof_stats=_proof_stats,
        text_report=_text_report
    )
