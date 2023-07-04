from multi_type_search.search.search_model.types.contrastive import ContrastiveModel

from copy import deepcopy
from simcse import SimCSE
import torch
from tqdm import tqdm
from functools import partialmethod
from pathlib import Path
from typing import List, Union, Dict, Tuple


class XCSE(ContrastiveModel):
    model_type: str = 'x_cse_contrastive_model'

    cse_type: str
    model_name: str
    device: str

    model: Union[SimCSE]

    def __init__(
            self,
            cse_type: str,
            model_name: str,
            device: str = 'cuda:0'
    ):
        super().__init__()

        self.cse_type = cse_type
        self.model_name = model_name
        self.device = device

        if cse_type == 'SimCSE':
            self.model = SimCSE(self.model_name)
        else:
            raise Exception(f"Unknown cse type {cse_type}!")

        self.encoding_tokenizer = self.model.tokenizer
        self.encoding_model = self.model.model.cuda()
        
        self.cuda()

        test = self.get_encodings(['Hi'])
        self.embedding_size = test.shape[-1]

    def get_kwargs(self):
        return {
            'cse_type': self.cse_type,
            'model_name': self.model_name,
            'device': self.device
        }

    def save(self, path: Path, save_backbone: bool = False):
        if save_backbone:
            torch.save({
                'kwargs': self.get_kwargs(),
                'backbone_state_dict': self.model.model.state_dict(),
                'type': self.model_type
            }, str(path))
        else:
            torch.save({
                'kwargs': self.get_kwargs(),
                'type': self.model_type
            }, str(path))

    def get_encodings(self, strings: List[str]) -> torch.Tensor:
        # TODO - hacks to remove the tqdm progress bar from the inner call.
        orig_init = deepcopy(tqdm.__init__)
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
        encodings = self.custom_encode(strings)
        tqdm.__init__ = orig_init
        return encodings

    def tokenize(self, exs: List[str]):
        max_length = 128

        inputs = self.encoding_tokenizer(
            exs,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        return {k: v.squeeze().cuda() if len(v.shape) > 2 else v for k, v in inputs.items()}

    def forward(self, tokens):
        return self.encode(tokens)

    def encode(self, tokens):
        outputs = self.encoding_model(**tokens, return_dict=True)
        if self.model.pooler == "cls":
            embeddings = outputs.pooler_output
        elif self.model.pooler == "cls_before_pooler":
            embeddings = outputs.last_hidden_state[:, 0]
        else:
            raise NotImplementedError

        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
        return embeddings

    def custom_encode(
            self,
            sentence: Union[str, List[str]],
            device: str = None,
            return_numpy: bool = False,
            normalize_to_unit: bool = True,
            keepdim: bool = False,
            batch_size: int = 64,
            max_length: int = 128
    ):
        # target_device = self.device if device is None else device
        # self.model.model = self.model.model.to(target_device)

        single_sentence = False
        if isinstance(sentence, str):
            sentence = [sentence]
            single_sentence = True

        embedding_list = []
        total_batch = len(sentence) // batch_size + (1 if len(sentence) % batch_size > 0 else 0)
        for batch_id in tqdm(range(total_batch)):
            _inputs = self.encoding_tokenizer(
                sentence[batch_id * batch_size:(batch_id + 1) * batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            inputs = {k:  (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in _inputs.items()}
            outputs = self.encoding_model(**inputs, return_dict=True)
            if self.model.pooler == "cls":
                embeddings = outputs.pooler_output
            elif self.model.pooler == "cls_before_pooler":
                embeddings = outputs.last_hidden_state[:, 0]
            else:
                raise NotImplementedError
            if normalize_to_unit:
                embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
            embedding_list.append(embeddings)
        embeddings = torch.cat(embedding_list, 0)

        if single_sentence and not keepdim:
            embeddings = embeddings[0]

        if return_numpy and not isinstance(embeddings, ndarray):
            return embeddings.numpy()
        return embeddings

    @classmethod
    def __load__(cls, data: Dict, device: str) -> 'XCSE':
        kwargs = data.get('kwargs')
        assert kwargs is not None, f'Error loading XCSE from checkpoint: {ckpt}, no kwargs in file.'
        if 'device' in kwargs:
            del kwargs['device']
        model = cls(**kwargs)
        model.model.model = model.model.model.to(device)
        model = model.cuda()
        #model.model.cuda()
        model.model.model = model.model.model.cuda()
        model.encoding_model = model.encoding_model.cuda()

        if 'backbone_state_dict' in data:
            state_dict = data.get('backbone_state_dict')
            model.model.model.load_state_dict(state_dict)

        return model
