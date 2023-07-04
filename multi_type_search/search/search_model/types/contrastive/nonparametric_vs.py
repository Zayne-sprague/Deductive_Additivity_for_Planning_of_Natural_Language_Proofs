from multi_type_search.search.search_model.types.contrastive import ContrastiveModel

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import RobertaTokenizer, RobertaModel, AutoModel, AutoConfig

import torch
from torch import nn
from torch.nn import functional as F
import transformers
from pathlib import Path
from typing import List, Union, Dict, Tuple, Optional


class NonParametricVectorSpace(ContrastiveModel):

    model_type: str = 'non_parametric_vector_space_contrastive_model'

    contrastive_model: Optional['ContrastiveModel']

    def __init__(
            self,
            simcse_name: str = None,
            t5_model_name: str = 't5-small',
            roberta_base: bool = False,
            t5_token_max_length: int = 128,
            cnn_hidden_size: int = 1024,
            fw_hidden_size=128,
            embedding_size=16,
            residual_connections: bool = False,
            backbone_only: bool = False,
            freeze_backbone: bool = True,
            device='cpu'
    ):
        super().__init__()

        from multi_type_search.search.search_model.types.contrastive import XCSE

        self.backbone_only = backbone_only
        self.freeze_backbone = freeze_backbone
        self.contrastive_model = None
        self.simcse_name = simcse_name
        self.t5_model_name = t5_model_name
        self.t5_token_max_length = t5_token_max_length
        self.residual_connections = residual_connections
        self.device = device
        self.fw_hidden_size = fw_hidden_size * 2
        self.embedding_size = embedding_size * 2

        self.using_roberta = roberta_base

        if roberta_base:

            self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base', model_max_length=self.t5_token_max_length)
            self.roberta_model = AutoModel.from_pretrained('roberta-base')
            config = AutoConfig.from_pretrained('roberta-base')
            self.roberta_projection = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps))
            self.init_encoding_size = 768
            self.roberta_model.cuda()
            self.roberta_projection.cuda()

            for p in self.roberta_model.parameters():
                p.requires_grad = True #not self.freeze_backbone

        elif simcse_name:
            self.contrastive_model = XCSE('SimCSE', simcse_name, device)
            self.init_encoding_size = self.contrastive_model.embedding_size

            for p in self.contrastive_model.encoding_model.parameters():
                p.requires_grad = not self.freeze_backbone
        else:
            self.t5tokenizer = T5Tokenizer.from_pretrained(t5_model_name, model_max_length=self.t5_token_max_length)
            self.t5model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

            self.init_encoding_size = self.t5model.model_dim

            # Freeze the t5 backbone.
            for p in self.t5model.parameters():
                p.requires_grad = not self.freeze_backbone

        if not self.backbone_only:


            self.fw = nn.Linear(self.init_encoding_size , self.fw_hidden_size)
            self.fw1_2 = nn.Linear(int(self.fw_hidden_size / 2) + (self.init_encoding_size if self.residual_connections else 0), self.fw_hidden_size)
            self.fw2 = nn.Linear(int(self.fw_hidden_size) + (self.init_encoding_size if self.residual_connections else 0), self.embedding_size)


            self.dd1 = nn.Dropout(p=0.1)
            self.dd2 = nn.Dropout(p=0.1)
            self.dd2_1 = nn.Dropout(p=0.1)
            self.dd3 = nn.Dropout(p=0.1)


    def get_kwargs(self):
        return {
                'simcse_name': self.simcse_name,
                't5_model_name': self.t5_model_name,
                't5_token_max_length': self.t5_token_max_length,
                'fw_hidden_size': self.fw_hidden_size,
                'embedding_size': self.embedding_size,
                'residual_connections': self.residual_connections,
                'device': self.device,
                'freeze_backbone': self.freeze_backbone,
                'backbone_only': self.backbone_only
            }

    def forward(self, tokens: Union[torch.Tensor, List[str]]):
        if isinstance(tokens, list) and isinstance(tokens[0], str):
            tokens = self.tokenize(tokens)
            if torch.cuda.is_available():
                tokens = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in tokens.items()}
            else:
                tokens = {k: v if isinstance(v, torch.Tensor) else v for k, v in tokens.items()}

        encodings = self.encode(tokens)

        if self.backbone_only:
            return encodings, encodings, encodings

        e = encodings

        e = self.dd1(e)

        h1 = self.fw(e)
        h1 = F.glu(h1, dim=1)

        # h1 = F.leaky_relu(h1)
        h1_raw = self.dd2(h1)

        if self.residual_connections:
            h1 = torch.cat([h1_raw, e], dim=1)
        else:
            h1 = h1_raw

        h1_2 = self.fw1_2(h1)
        h1_2 = F.glu(h1_2, dim=1) 

        h1_2 = self.dd2_1(h1_2)

        if self.residual_connections:
            h1_2 = torch.cat([h1_2, h1_raw, e], dim=1)

        h2 = self.fw2(h1_2)

        # h2 = torch.cat([h2, h1, e], dim=1)
        h2 = F.glu(h2, dim=1)

        return h2, h1, encodings

    def tokenize(self, exs: Union[List[str], str]):
        if isinstance(exs, str):
            exs = [exs]

        if self.using_roberta:
            #return {k: v.squeeze() if len(v.shape) > 2 else v for k, v in self.roberta_tokenizer(exs, return_tensors="pt", max_length=self.t5_token_max_length, padding='max_length', truncation=True).items()}
            return self.roberta_tokenizer(exs, return_tensors="pt", max_length=self.t5_token_max_length, padding='max_length', truncation=True)
        elif self.contrastive_model:
            return self.contrastive_model.tokenize(exs)
        return {k: v.squeeze() if len(v.shape) > 2 else v for k, v in self.t5tokenizer(exs, return_tensors="pt", max_length=self.t5_token_max_length, padding='max_length', truncation=True).items()}

    def encode(self, tokens):
        if self.using_roberta:
            if torch.cuda.is_available():
                tokens = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in tokens.items()}
            else:
                tokens = {k: v if isinstance(v, torch.Tensor) else v for k, v in tokens.items()}

            encs = self.roberta_projection(self.roberta_model(**tokens)[0][:, 0, :])
        elif self.contrastive_model:
            encs = self.contrastive_model.encode(tokens)
        else:
            encs = self.t5model.encoder(**tokens)
            encs = encs.last_hidden_state
            encs = torch.permute(encs, (0, 2, 1))
            encs = torch.nn.functional.avg_pool1d(encs, (encs.shape[-1]))
            if len(encs.shape) > 2:
                encs = encs.squeeze(-1)
        return encs


    def init_encode_exs(self, exs: Union[List[str], str]) -> torch.Tensor:
        if isinstance(exs, str):
            exs = [exs]

        if self.contrastive_model:
            return self.contrastive_model.get_encodings(exs)

        input_ids = [
            self.t5tokenizer(ex, return_tensors="pt", max_length=self.t5_token_max_length, padding='max_length', truncation=True).input_ids
            for ex in exs]
        encodings = [self.t5model.encoder(input_ids=ex_ids).last_hidden_state for ex_ids in input_ids]

        mask = (torch.cat(input_ids, dim=0) != 0).unsqueeze(-1)
        zeroed_encodings = (mask * torch.cat(encodings, dim=0))

        pooled_encodings = zeroed_encodings.sum(1) / (mask.sum(1) + 1e-5)

        return pooled_encodings

    def get_encodings(self, strings: List[str]):
        o, _, _ = self(self.tokenize(strings))
        return o

    @classmethod
    def __load__(cls, data: Dict, device: str, opt) -> 'NonParametricVectorSpace':
        kwargs = data.get('kwargs')
        assert kwargs is not None, f'Error loading node embedder from checkpoint: {ckpt}, no kwargs in file.'

        if 'device' in kwargs:
            del kwargs['device']
        if 'embedding_size' in kwargs:
            kwargs['embedding_size'] = int(kwargs['embedding_size'] / 2)
        if 'fw_hidden_size' in kwargs:
            kwargs['fw_hidden_size'] = int(kwargs['fw_hidden_size'] / 2)
        if 'opt_state' in data and opt:
            opt.load_state_dict(data['opt_state'])

        model = cls(**kwargs, device=device, roberta_base=True)

        state_dict = data.get('state_dict')
        assert state_dict is not None, f'Error loading node embedder from checkpoint: {ckpt}, no state dict in file.'
        model.load_state_dict(state_dict, strict=False)

        model.to(device)
        model.device = device

        if model.contrastive_model:
            model.contrastive_model.to(device)
            
        #model.t5model.to(device)
        if opt:
            return model, opt

        return model

