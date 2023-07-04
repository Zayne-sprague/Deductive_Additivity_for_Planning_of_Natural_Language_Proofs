from pathlib import Path
import torch
from torch import nn
from abc import ABC, abstractmethod, abstractclassmethod
from typing import List, Dict


class ContrastiveModel(nn.Module, ABC):
    """
    Base class for all contrastive based models that encode strings into a fixed vector
    representation. (Used mostly in NodeEmbedder)
    """

    embedding_size: int

    @classmethod
    def load(cls, path: Path, device: str, opt = None) -> 'ContrastiveModel':
        from multi_type_search.search.search_model.types.contrastive import NonParametricVectorSpace, \
            WordToVec, XCSE, RobertaEncoder, RobertaMomentumEncoder, RawGPT3Encoder, ProjectedGPT3Encoder  #, Glove

        data = torch.load(str(path), map_location=torch.device(device))

        # TODO - default is to support previously trained models, should be removed
        model_type = data.get('type', NonParametricVectorSpace.model_type)

        if model_type == NonParametricVectorSpace.model_type:
            return NonParametricVectorSpace.__load__(data, device, opt)
        if model_type == WordToVec.model_type:
            return WordToVec.__load__(data, device)
        if model_type == XCSE.model_type:
            return XCSE.__load__(data, device)
        # if model_type == Glove.model_type:
        #     return Glove.__load__(data, device)
        if model_type == RobertaEncoder.model_type:
            return RobertaEncoder.__load__(data, device, opt)
        if model_type == RobertaMomentumEncoder.model_type:
            return RobertaMomentumEncoder.__load__(data, device, opt)
        if model_type == RawGPT3Encoder.model_type:
            return RawGPT3Encoder.__load__(data, device, opt)
        if model_type == ProjectedGPT3Encoder.model_type:
            return ProjectedGPT3Encoder.__load__(data, device, opt)

        raise Exception(f"Attempted to load an unknown contrastive model {model_type} from {path}.")

    def save(self, path: Path, opt = None):
        data = {
            'kwargs': self.get_kwargs(),
            'state_dict': self.state_dict(),
            'type': self.model_type,
        }
        if opt:
            data['opt_state'] = opt.state_dict()
        torch.save(data, str(path))

    @abstractmethod
    def get_encodings(self, strings: List[str]) -> torch.Tensor:
        raise NotImplemented("Implement get_encodings() for all contrastive models!")

    @abstractmethod
    def get_kwargs(self) -> Dict:
        raise NotImplemented("Implement get_kwargs() for all contrastive models!")

    @classmethod
    @abstractmethod
    def __load__(cls, data: Dict, device: str) -> 'ContrastiveModel':
        raise NotImplemented("Implement __load__() for all contrastive models!")
