from multi_type_search.search.search_model.types.contrastive import ContrastiveModel

import gensim
from gensim.models.word2vec import Word2Vec, KeyedVectors
import torch
from torch import nn
from pathlib import Path
from typing import List, Union, Dict, Tuple


class WordToVec(ContrastiveModel):
    model_type: str = 'word_to_vec_contrastive_model'

    device: str
    vocab: Dict
    embedding_size: int

    def __init__(
            self,
            vocab: Dict,
            embedding_size: int,
            device: str = 'cpu'
    ):
        super().__init__()
        self.vocab = vocab
        self.embedding_size = embedding_size
        self.device = device

        self.emb = nn.Embedding(len(list(self.vocab.keys())), embedding_size, padding_idx=0).to(device)

    def get_kwargs(self):
        return {
            'vocab': self.vocab,
            'embedding_size': self.embedding_size,
            'device': self.device
        }

    def forward(self, x: torch.Tensor):
        return self.emb(x)

    def get_encodings(self, strings: List[str]) -> torch.Tensor:
        encodings = []
        for string in strings:
            tokens = string.split()
            ids = [self.vocab.get(x, 0) for x in tokens]
            unks = sum([1 if x == 0 else 0 for x in ids])

            if len(ids) - unks == 0:
                encodings.append(torch.zeros(self.embedding_size))
                continue

            encodings.append(self(torch.tensor(ids).to(self.device)).sum(0) / (len(ids) - unks))

        return torch.stack(encodings).squeeze(1).to(self.device)

    @classmethod
    def __load__(cls, data: Dict, device: str) -> 'WordToVec':
        kwargs = data.get('kwargs')
        assert kwargs is not None, f'Error loading word to vec from checkpoint: {ckpt}, no kwargs in file.'
        model = cls(**kwargs)

        state_dict = data.get('state_dict')
        assert state_dict is not None, f'Error loading node embedder from checkpoint: {ckpt}, no state dict in file.'
        model.load_state_dict(state_dict)

        model.to(device)
        model.emb.to(device)
        model.device = device

        return model
