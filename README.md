# Deductive Additivity for Planning of Natural Language Proofs


<image src="./figs/diagram.png"></image>

Vector-based planning of natural language reasoning

## Overview

This repository holds the code for the paper _Deductive Additivity for Planning of Natural Language Proofs_

All the data for training and evaluating models can be found in `{PROJECT_ROOT}/data`

The newly introduced **Single Step Reasoning Contrast** dataset can be found in `{PROJECT_ROOT}/data/full/reasoning_benchmarks/benchmark.csv`

## Installation

1. `virtualenv venv` we have tested with python 3.8
2. `source venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python ./install.py` (expect this to take a long time based on your internet speed)
5. `export PYTHONPATH=$PWD`

Step 4 will download all the models needed to recreate the experiments from the paper as well as two small step models
that also perform well.  (We use the smaller step models for testing/debugging and the 3billion parameter models for
experiments.)

You can see all the models that can be downloaded via
`python ./install.py -s` 

And you can install them individually via
`python ./install.py -a abductive_heuristic abductive_step_model_small ...etc`

## Recreating Paper Results

A few things about running experiments.

All experiments should have a config file that helps specify their parameters, denoted by the `-c` flag.  These config files can be found in `multi_type_search/configs/`.

The existing config files should match what was tested in the paper, bm25, gpt3_raw, gpt3_trained, scsearch, and simcse.

Most experiments will have an output flag `-o` which takes in a subpath to use to store all information about the experiment.  You can find the output in `multi_type_search/output/{-o flag subpath}`.  You can specific `-f` to overwrite existing subpaths, otherwise it will fail if an existing folder is already there.

### Intrinsic Embedding Representations

Go into the `multi_step_type/scripts` folder.

```shell
python heuristic_emb_rep_benchmark.py -c heuristic_benchmarks/gpt3_raw -o output_subpath -f
```

Similarily, to measure the MRR of a heuristic:

```shell
python heuristic_mrr_benchmark.py -c heuristic_benchmarks/gpt3_raw -o output_subpath -f
```

Inside the config files, you can configure each of these experiments under their respective yaml sections (rep_benchmark and mrr_benchmark respectively.)  These changes include what model, model hyperparameters, and datasets.

### Proof Generation

Go into the `multi_type_search/experiemnts` folder and run:

```shell
python search_experiment.py -cn heuristics/raw_gpt3 -en output_subpath -f -rc
```

These config files are separate from the intrensic experiment config files due to them mostly being specific to the paper [_Natural Language Deduction with Incomplete Information_](https://arxiv.org/abs/2211.00614).  You can read more about how these config files work [here](https://github.com/Zayne-sprague/Natural_Language_Deduction_with_Incomplete_Information).

### SSRC Dataset MRR

Go into the `multi_type_search/scripts` folder and run:

```shell
python heuristic_mrr_reasoning_benchmark.py -c heuristic_benchmarks/gpt3_raw -o output_subpath -f
 ```

The configuration for this experiment can be found under the heading `mrr_reasoning_benchmark` in the config files.

### Training a contrastive model like "GPT3-Trained"

Go into the folder `multi_type_search/scripts/contrastive` and run

```shell
python train_contrastive_model.py -lf NTXENTLoss -tss -tf ../../../data/full/entailmentbank/task_1/train.jsonl -vf ../../../data/full/entailmentbank/task_2/test.jsonl -gcef data/full/entailmentbank/ebt1_tr_ts_ebt2_ts/embeddings.pkl -gcsf data/full/entailmentbank/ebt1_tr_ts_ebt2_ts/strings.json -tau 0.07 -r new_model_name -t -dmv -tcm add -hmn projected_gpt3 -bs 100 -gphln 3 -gpht glu -tplw 1.0
```

There are a ton of hyperparameters that were created for training this model.  The most important will be defined here, but you can open the training script and find definitions for each (or use the -h flag).

`-tf` - the training file path relative to your cwd.

`-vf` - the validation file path relative to your cwd.

`-gcef` - A cached file of gpt3 raw embeddings for all training and validation premises and deductions (so we do not have to query them while training which is slow, and expensive)

`-gcsf` - A cached file of gpt3 raw strings matching to the embeddings file (same order).  These are created by `multi_type_search/scripts/embed_with_gpt3` (you will have to add your api key to the script).

`-tau` - temperature of the NTXENTLoss. 

`-r` - name of the run, the model will be saved under `trained_models/{-r}`

`-t` - Train

`-dmv` - Do MRR validation (MRR will be reported every epoch on the validation set)

`-tcm` - Trajectory creation method, we explored more than just addition there is also subtract, multiply, max_pool, min_pool, and avg_pool.

`-bs` - Batch size

`-gphln` - Number of the gpt3 projection layers.

`-gpht` - GPT3 projection head type (glu or linear, effects the activation functions)

`-tplw` - Two premise loss weight.  We explored a variety of losses, we found that this one does the best solely.

Many many more are in the training file.  If anyone is interested in any of the specifics, don't hesitate to reach out!


## Requirements
All experiments were performed on the following specs (some may not matter)

- Python Version: 3.8.3
- tested on Mac and Linux
- Transformers version 4.20.0 (really important for proof generation)
    - Transformers version 4.10.4 (really important for intrinsic eval)

When you want to run the intrensic evaluations (MRRs etc.) use `pip install -r requirements.txt`

If you want to generate proof trees use `pip install -r requirements_proof_gen_exp.txt`

### Authors
- Zayne Sprague
- Kaj Bostrom
- Swarat Chaudhuri
- Greg Durrett


