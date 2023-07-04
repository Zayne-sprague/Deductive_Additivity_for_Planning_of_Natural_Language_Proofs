import json

import torch
from torch import optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn import Module
from jsonlines import jsonlines
from pathlib import Path
from typing import List, Dict

from multi_type_search.scripts.contrastive.contrastive_dataset import ContrastiveDataset
from multi_type_search.utils.paths import DATA_FOLDER
from multi_type_search.search.graph import Graph
from multi_type_search.search.search_model.types.contrastive import ContrastiveModel, NonParametricVectorSpace
from multi_type_search.search.search_model.types.contrastive.contrastive_utils import cosine_similarity_metric
from multi_type_search.scripts.contrastive.train_contrastive_model import train_epoch

import streamlit as st

training_dataset = None
validation_dataset = None
model = None


def inference(strings):
    global model
    toks = model.tokenize(strings)
    encs, _, _ = model(toks)
    return encs


def dataset_ex_inference(ex):
    strings = [*ex[1], *ex[2], *ex[3]]
    encs = inference(strings)

    trajectory = encs[0] + encs[1]
    deductive_score = cosine_similarity_metric(trajectory, encs[2]).item()
    goal_score = cosine_similarity_metric(trajectory, encs[3]).item()
    return deductive_score, goal_score


def training_dataset_loader():
    with st.expander("Training Data"):
        training_dataset_file = st.file_uploader(
            "Training dataset", ['json', 'jsonl'], accept_multiple_files=True,
            help='Json or Jsonl file that contains Graphs with deductions that can be used as training data for the contrastive'
                 'model.',
        )

        if not training_dataset_file:
            st.stop()

        graphs = []
        for file in training_dataset_file:
            graphs.extend(
                [Graph.from_json(x) for x in (json.load(file) if file.name.endswith('.json') else jsonlines.Reader(file))])

        training_dataset = ContrastiveDataset()
        training_dataset.populate_examples(graphs)

        dataset_number = st.number_input('See example in training dataset', min_value=0,
                                         max_value=len(training_dataset) - 1)
        ex = training_dataset[dataset_number]
        args = '\n'.join(ex[1])

        st.write(f"Dataset size: {len(training_dataset) - 1}")
        st.subheader(f"Example {dataset_number}:")
        st.write(f"Graph number: {ex[0]}")
        st.write("Arguments: ")
        st.code(args, language=None)
        st.write("Deductive Goal: ")
        st.code(ex[2], language=None)
        st.write("Graph Goal: ")
        st.code(ex[3], language=None)

        return ex, training_dataset


def validation_dataset_loader():
    global model, training_dataset

    with st.expander("Validation data"):
        use_training_as_validation = st.checkbox('Use Training data as validation', value=False)

        if use_training_as_validation:
            validation_dataset = training_dataset
            ex = None
        else:
            validation_dataset_file = st.file_uploader(
                "Validation dataset", ['json', 'jsonl'], accept_multiple_files=True,
                help='Json or Jsonl file that contains Graphs with deductions that can be used as validation data for the '
                     'contrastive model.',
            )

            if not validation_dataset_file:
                st.stop()

            graphs = []
            for file in validation_dataset_file:
                graphs.extend(
                    [Graph.from_json(x) for x in (json.load(file) if file.name.endswith('.json') else jsonlines.Reader(file))])

            validation_dataset = ContrastiveDataset()
            validation_dataset.populate_examples(graphs)

            dataset_number = st.number_input('See example in validation dataset', min_value=0,
                                             max_value=len(validation_dataset) - 1)
            ex = validation_dataset[dataset_number]
            args = '\n'.join(ex[1])

            st.write(f"Dataset size: {len(validation_dataset) - 1}")
            st.subheader(f"Example {dataset_number}:")
            st.write(f"Graph number: {ex[0]}")
            st.write("Arguments: ")
            st.code(args, language=None)
            st.write("Deductive Goal: ")
            st.code(ex[2], language=None)
            st.write("Graph Goal: ")
            st.code(ex[3], language=None)

        return ex, validation_dataset


def new_model_form():
    backbone = st.radio("Backbone", options=['t5', 'SIMCSE'])
    backbone_only = st.checkbox('Backbone Only', value=False, help='No extra layers will be added to the model.')

    with st.form("new_model_form"):
        model_name = st.text_input("Model Name")

        t5_model_name = 't5-small'
        t5_token_max_length = 128
        simcse_name = None
        fw_hidden_size = 1024
        embedding_size = 512
        residual_connections = False

        if backbone == 't5':
            t5_model_name = f"t5-{st.selectbox('T5 Model Select', ['small', 'base', 'large'])}"
            t5_token_max_length = st.number_input(label='Max Token Length', min_value=0, max_value=None, value=128, step=1)
        else:
            simcse_name = st.selectbox('SIMCSE Model Select', [
                'princeton-nlp/sup-simcse-roberta-base',
                'princeton-nlp/sup-simcse-bert-base-uncased',
                'princeton-nlp/sup-simcse-roberta-large',
                'princeton-nlp/unsup-simcse-roberta-large',
                'princeton-nlp/sup-simcse-bert-large-uncased',
                'princeton-nlp/unsup-simcse-bert-large-uncased',
            ], index=0)


        if not backbone_only:
            fw_hidden_size = st.number_input('Forward Layer Hidden Size', value=1024)
            embedding_size = st.number_input('Embedding Size', value=512)
            residual_connections = st.checkbox('Residual Connections', value=True, help='Whether previous custom layers should being concatinated together.')

        freeze_backbone = st.checkbox('Freeze Backbone', value=True, help='Allow the backbone model to be trained')

        if backbone is None:
            st.stop()

        submitted = st.form_submit_button("Submit")
        if submitted:

            return model_name, {
                'simcse_name': simcse_name, 't5_model_name': t5_model_name, 't5_token_max_length': t5_token_max_length,
                'fw_hidden_size': fw_hidden_size, 'embedding_size': embedding_size,
                'residual_connections': residual_connections, 'backbone_only': backbone_only, 'freeze_backbone': freeze_backbone
            }
        else:
            st.stop()


def basic_model_args() -> NonParametricVectorSpace:
    with st.expander("Basic Model Arguments"):

        if torch.cuda.is_available():
            device_type = st.radio('device type', options=['GPU', 'CPU'])
            if device_type == 'CPU':
                device = "cpu"
            else:
                cuda_devices = st.multiselect("GPU Devices", options=list(range(torch.cuda.device_count() - 1)))
                device = cuda_devices[0]
        else:
            device = "cpu"

        load_option = st.radio("New model or load existing checkpoint", options=["New", "Load"])

        if load_option == 'New':

            model_name, args = new_model_form()
            with st.spinner("Building Model..."):
                model = NonParametricVectorSpace(**args)


        else:

            model_file = st.file_uploader(
                'Load the file checkpoint', type='.pth', accept_multiple_files=False,
                help='Select the checkpoint file for the model you wish to load.'
            )

            if model_file is None:
                st.stop()

            with st.spinner("Loading model..."):
                model_data = torch.load(model_file, map_location=torch.device(device))
                model = NonParametricVectorSpace.__load__(model_data, device)

        return model

def example_inferences(training_ex, validation_ex, model):
    if training_ex is not None and st.button("Model Training Inference"):
        with st.spinner('Inference running...'):
            if model is None:
                st.warning('Need to load a model first.')
                st.stop()
            deductive_score, goal_score = dataset_ex_inference(training_ex)
            st.write(f"Deductive score: {deductive_score:.4f}")
            st.write(f"Goal score: {goal_score:.4f}")

    if validation_ex is not None and st.button("Model Validation Inference"):
        with st.spinner('Inference running...'):

            if model is None:
                st.warning('Need to load a model first.')
                st.stop()
            deductive_score, goal_score = dataset_ex_inference(validation_ex)
            st.write(f"Deductive score: {deductive_score:.4f}")
            st.write(f"Goal score: {goal_score:.4f}")


def advanced_training_params(model):
    loss_fn = None
    optimizer = None

    batch_size = st.number_input('Batch size', min_value=1, value=512, step=1)
    learning_rate = st.number_input('Learning Rate', min_value=0., max_value=1000., value=0.001, step=1e6)

    weight_decay = st.number_input('Weight Decay', min_value=0., value=0., step=1e3)
    momentum = st.number_input('Momentum', min_value=0., value=0., step=1e3)

    optimizer_args = {
        'lr': learning_rate
    }

    if weight_decay > 0:
        optimizer_args['weight_decay'] = weight_decay
    if momentum > 0:
        optimizer_args['momentum'] = momentum


    optimizer_name = st.selectbox('Optimizer', [
        'Adam',
        'SGD',
        'RMSprop'
    ])

    max_epochs = st.number_input('Max Epochs', min_value=1, max_value=None, value=2000, step=1)

    loss_fn = st.selectbox('Loss Function', options=[
        'NTXentloss',
        'Cosine Sim',
        'MSE'
    ])

    deductive_loss_term_weight = st.number_input('Deductive Loss Term Weight', min_value=0, max_value=None, value=0.99, step=1e3)
    graph_goal_loss_term_weight = st.number_input('Graph Goal Loss Term Weight', min_value=0, max_value=None, value=0., step=1e3)
    condition_number_regularization = st.number_input('Condition Number Regularization Term Weight', min_value=0, max_value=None, value=0.01, step=1e3)

    if loss_fn == 'NTXentloss':
        ntxent_tau = st.number_input('NTXent Tau', min_value=0, max_value=None, value=0.05, step=0.001)

        loss_fn = NTXENTLoss(
            temperature=ntxent_tau
        )
    else:
        st.warning("TODO: Implement these!")
        pass

    optimizer = getattr(optim, optimizer_name)
    optimizer = optimizer(
        model.parameters(recurse=True),
        **optimizer_args
    )

    return loss_fn, optimizer, deductive_loss_term_weight, graph_goal_loss_term_weight, condition_number_regularization, max_epochs, batch_size


def handle_train_epoch(
        model: NonParametricVectorSpace,
        training_dataset: ContrastiveDataset,
        batch_size: int,
        optimizer: Optimizer,
        loss_fn: Module,
        deductive_loss_term_weight: float = 1.0,
        graph_goal_loss_term_weight: float = 0.0,
        condition_number_regularization_term_weight: float = 0.0,
):

    if st.button("Train for Epoch"):

        dataloader = DataLoader(
            training_dataset, batch_size, True, collate_fn=training_dataset.collate_fn,
            pin_memory=True, num_workers=4
        )

        pbar = st.progress("Running epoch", value=0)
        total_batches = 0
        total_len = len(dataloader)

        def callback():
            pbar.progress((total_batches + 1) / total_len)

        train_epoch(
            model,
            dataloader,
            optimizer,
            loss_fn,
            deductive_loss_term_weight,
            graph_goal_loss_term_weight,
            condition_number_regularization_term_weight,
            callback
        )


st.write("""
# Training Contrastive models
""")

training_ex, training_dataset = training_dataset_loader()
validation_ex, validation_dataset = validation_dataset_loader()

model = basic_model_args()
example_inferences(training_ex, validation_ex, model)
loss_fn, optimizer, deductive_loss_term_weight, graph_goal_loss_term_weight, condition_number_regularization, max_epochs, batch_size = advanced_training_params(model)

handle_train_epoch(
    model, training_dataset, batch_size, optimizer, loss_fn,
    deductive_loss_term_weight, graph_goal_loss_term_weight, condition_number_regularization
)