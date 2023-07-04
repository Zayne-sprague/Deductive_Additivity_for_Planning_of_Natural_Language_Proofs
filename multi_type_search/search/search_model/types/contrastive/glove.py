"""
A lot of this code is taken from
"""

import torch
from torch import nn
import numpy as np
from pathlib import Path
from typing import List, Dict
from torchtext.vocab import GloVe
from torchtext.data import get_tokenizer

from multi_type_search.search.search_model.types.contrastive import ContrastiveModel


class Glove(ContrastiveModel):
    model_type: str = 'glove_contrastive_model'

    device: str
    embedding_size: int
    name: str
    tokenizer_name: str

    def __init__(
            self,
            name: str = '840B',
            embedding_size: int = 300,
            tokenizer_name: str = 'basic_english',
            device: str = 'cpu'
    ):
        super().__init__()

        self.name = name
        self.embedding_size = embedding_size
        self.tokenizer_name = tokenizer_name
        self.device = device

        self.glove = GloVe(name=name, dim=embedding_size)
        self.tokenizer = get_tokenizer(tokenizer_name)

    def get_kwargs(self):
        return {
            'name': self.name,
            'embedding_size': self.embedding_size,
            'tokenizer_name': self.tokenizer_name,
            'device': self.device
        }

    def get_encodings(self, strings: List[str]) -> torch.Tensor:
        encodings = []
        for string in strings:
            vecs = self.glove.get_vecs_by_tokens(self.tokenizer(string))
            encodings.append(vecs.sum(0) / vecs.shape[0])
        return torch.stack(encodings).squeeze(1).to(self.device)

    @classmethod
    def __load__(cls, data: Dict, device: str) -> 'Glove':
        kwargs = data.get('kwargs')
        assert kwargs is not None, f'Error loading word to vec from checkpoint: {ckpt}, no kwargs in file.'
        model = cls(**kwargs)

        model.to(device)
        return model
