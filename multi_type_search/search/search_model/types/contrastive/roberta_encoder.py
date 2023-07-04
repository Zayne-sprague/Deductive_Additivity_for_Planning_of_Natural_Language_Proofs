from multi_type_search.search.search_model.types.contrastive import ContrastiveModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import RobertaTokenizer, RobertaModel, AutoModel, AutoConfig, AutoTokenizer

import torch
from torch import nn
from torch.nn import functional as F
import transformers
from pathlib import Path
from typing import List, Union, Dict, Tuple, Optional


class RobertaEncoder(ContrastiveModel):
    model_type: str = 'roberta_encoder'

    def __init__(
            self,
            roberta_model_name: str = 'roberta-base',
            max_token_length: int = 256
    ):
        super().__init__()

        self.max_token_length = max_token_length
        self.roberta_model_name = roberta_model_name

        self.roberta_tokenizer = AutoTokenizer.from_pretrained(
            roberta_model_name,
            model_max_length=max_token_length
        )
        self.roberta_model = AutoModel.from_pretrained(roberta_model_name)
        self.config = AutoConfig.from_pretrained(roberta_model_name)

        self.roberta_projection = nn.Sequential(nn.Linear(self.config.hidden_size, self.config.hidden_size),
                                                nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps))

        self.head = nn.Sequential(
            nn.Linear(self.config.hidden_size * 2, self.config.hidden_size),
            nn.LayerNorm(self.config.hidden_size, eps=self.config.hidden_size),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.LayerNorm(self.config.hidden_size, eps=self.config.hidden_size),
                nn.ReLU()
            ) for _ in range(2 - 2)],
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.LayerNorm(self.config.hidden_size, eps=self.config.hidden_size)
        )

    def get_kwargs(self):
        return {
                'roberta_model_name': self.roberta_model_name,
                'max_token_length': self.max_token_length,
            }

    def tokenize(self, exs: Union[List[str], str]):
        if isinstance(exs, str):
            exs = [exs]
        return self.roberta_tokenizer(exs, return_tensors="pt", max_length=self.max_token_length, padding='max_length', truncation=True)

    def forward(self, tokens: Union[torch.Tensor, List[str]]):
        if isinstance(tokens, list) and isinstance(tokens[0], str):
            tokens = self.tokenize(tokens)
            if torch.cuda.is_available():
                tokens = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in tokens.items()}
            else:
                tokens = {k: v if isinstance(v, torch.Tensor) else v for k, v in tokens.items()}

        encodings = self.roberta_projection(self.roberta_model(**tokens)[0][:, 0, :])
        # encodings = self.roberta_model(**tokens)[0][:, 0, :]
        return encodings

    def get_encodings(self, strings: List[str]) -> torch.Tensor:
        return self(strings)

    def encode_pair(self, x):
        assert len(x.shape) == 3, 'Input shape expected [Batch, Pairs, Embeddings]'

        # x = torch.cat(x, dim=1)
        x = x.reshape(x.shape[0], -1)
        rep = self.head(x)
        return rep

    @classmethod
    def __load__(cls, data: Dict, device: str, opt) -> 'RobertaEncoder':
        kwargs = data.get('kwargs')
        assert kwargs is not None, f'Error loading node embedder from checkpoint: {ckpt}, no kwargs in file.'

        if 'opt_state' in data and opt:
            opt.load_state_dict(data['opt_state'])

        model = cls(**kwargs)

        state_dict = data.get('state_dict')
        assert state_dict is not None, f'Error loading node embedder from checkpoint: {ckpt}, no state dict in file.'
        model.load_state_dict(state_dict, strict=False)

        model.to(device)

        return model


class RobertaMomentumEncoder(ContrastiveModel):
    model_type: str = 'roberta_momentum_encoder'

    def __init__(self, model: RobertaEncoder, queue_size: int = 2048, use_momentum_encoder: bool = True):
        super().__init__()
        self.queue_size = queue_size
        self.use_momentum_encoder = use_momentum_encoder

        self.q_enc = model

        if self.use_momentum_encoder and False:
            self.k_enc = RobertaEncoder(**model.get_kwargs())
            for param_q, param_k in zip(self.q_enc.parameters(), self.k_enc.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
        else:
            self.k_enc = None

        self.roberta_tokenizer = self.q_enc.roberta_tokenizer

        self.register_buffer("arg1_queue", torch.randn(self.queue_size, self.q_enc.config.hidden_size))
        self.register_buffer("arg1_queue_ptr", torch.zeros(1, dtype=torch.long))
        #
        self.register_buffer("arg2_queue", torch.randn(self.queue_size, self.q_enc.config.hidden_size))
        self.register_buffer("arg2_queue_ptr", torch.zeros(1, dtype=torch.long))
        #
        self.register_buffer("query_queue", torch.randn(self.queue_size, self.q_enc.config.hidden_size))
        self.register_buffer("query_queue_ptr", torch.zeros(1, dtype=torch.long))

    def tokenize(self, exs: Union[List[str], str]):
        return self.q_enc.tokenize(exs)

    @torch.no_grad()
    def momentum_update(
            self,
            momentum: float = 0.999
    ):
        """
        Momentum encoder update, inspired by
        https://github.com/facebookresearch/multihop_dense_retrieval/blob/main/mdr/retrieval/models/mhop_retriever.py#L78

        :param momentum:
        :return:
        """

        assert self.use_momentum_encoder, \
            'This RobertaMomentumEncoder was created without the use_momentum_encoder flag'

        for param_q, param_k in zip(self.q_enc.parameters(), self.k_enc.parameters()):
            param_k.data = param_k.data * momentum + param_q * (1. - momentum)

    @torch.no_grad()
    def dequeue_and_enqueue(self, embeddings: torch.tensor, queue_name: str):
        """
        Memory bank enqueue and dequeue operation, inspired by
        https://github.com/facebookresearch/multihop_dense_retrieval/blob/main/mdr/retrieval/models/mhop_retriever.py#L86

        :param embeddings:
        :return:
        """

        batch_size = embeddings.shape[0]

        ptr = int(self.__getattr__(f'{queue_name}_ptr'))

        update_size = min([batch_size, self.queue_size - ptr])

        # If we have space to update the queue then update (enqueue and dequeue)
        if update_size > 0:
            self.__getattr__(queue_name)[ptr:ptr + update_size] = embeddings[:update_size]
            ptr += update_size

        # If our batch size exceeded the size of the queue, wrap back around to the front and dequeue/enqueue.
        if update_size < batch_size:
            self.__getattr__(queue_name)[0:batch_size - update_size] = embeddings[update_size:]
            ptr = batch_size - update_size

        self.__getattr__(f'{queue_name}_ptr')[0] = ptr

    def get_encodings(self, strings: List[str]) -> torch.Tensor:
        return self.q_enc(strings)

    def forward(self, x):
        return self.q_enc(x)

    def get_kwargs(self) -> Dict:
        return {
            'queue_size': self.queue_size,
            'use_momentum_encoder': self.use_momentum_encoder,
            **self.q_enc.get_kwargs(),
        }
    @classmethod
    def __load__(cls, data: Dict, device: str, opt) -> 'RobertaMomentumEncoder':
        kwargs = data.get('kwargs')
        assert kwargs is not None, f'Error loading node embedder from checkpoint: {ckpt}, no kwargs in file.'

        if 'opt_state' in data and opt:
            opt.load_state_dict(data['opt_state'])

        q_model = RobertaEncoder(roberta_model_name=kwargs.get('roberta_model_name', 'microsoft/roberta-base'), max_token_length=kwargs.get('make_token_length', 256))
        model = cls(q_model, kwargs.get('queue_size', 2048))

        state_dict = data.get('state_dict')

        assert state_dict is not None, f'Error loading node embedder from checkpoint: {ckpt}, no state dict in file.'
        model.load_state_dict(state_dict, strict=False)

        model.to(device)

        if opt:
            return model, opt
        return model

    def encode_pair(self, x):
        rep = self.q_enc.encode_pair(x)
        return rep

