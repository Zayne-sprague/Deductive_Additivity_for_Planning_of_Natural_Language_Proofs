import json

from multi_type_search.search.graph import Node, HyperNode, Graph
from multi_type_search.search.search_model import SearchModel
from multi_type_search.search.search_model.types.contrastive import ContrastiveModel
from multi_type_search.utils.paths import TRAINED_MODELS_FOLDER, ROOT_FOLDER

from datasets import Dataset, disable_progress_bar
disable_progress_bar()

import torch
from typing import List, Union, Dict, Tuple
from pathlib import Path
import numpy as np


class NodeEmbedder(SearchModel):
    """
    A Node Embedder can take a nodes value and convert it to some latent vector by taking the encoding of a T5 model and
    a custom transfer layer into the fixed size vector defined by the params in this model.
    """

    search_obj_type: str = 'node_embedder'

    model: ContrastiveModel
    batch_size: int
    device: str
    cached_tensors_file: str
    cached_index_map_file: str

    emb_cache: torch.tensor
    cache_map: Dict[str, int]


    def __new__(
            cls,
            model_name: str = '',
            batch_size: int = 4,
            device: str = 'cpu',
            cached_tensors_file: str = None,
            cached_index_map_file: str = None,
            force_new_instance: bool = False
    ):
        """ creates a singleton object, if it is not created, or else returns the previous singleton object"""

        instance_name = f'instance__{model_name}_{batch_size}_{device}'
        if force_new_instance:
            return super(NodeEmbedder, cls).__new__(cls)

        if not hasattr(cls, instance_name):
            setattr(cls, instance_name, super(NodeEmbedder, cls).__new__(cls))
        return getattr(cls, instance_name)

    def __init__(
            self,
            model_name: str = None,
            batch_size: int = 4,
            device: str = 'cpu',
            cached_tensors_file: str = None,
            cached_index_map_file: str = None,
            force_new_instance: bool = False
    ):
        """
        :param model_name: The name of the model (name of the folder in ROOT_PROJECT_DIRECTOR/trained_models)
        :param batch_size: Size of the batch for batched calls
        :param device: Torch device to run the model on
        :param cached_tensors_file: Cache of tensors for strings
        :param cached_index_map_file: Cache of string to idx for looking up strings in the cached_tensors_file
        :param force_new_instance: This class will try to use a singleton pattern so that you do not load the same model
            twice, you can skip this logic by turning this parameter to true.
        """

        if hasattr(self, 'instantiated'):
            return

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.torch_device = torch.device(device)

        if cached_index_map_file and cached_tensors_file:
            self.cached_tensors_file = ROOT_FOLDER / cached_tensors_file
            self.cached_index_map_file = ROOT_FOLDER / cached_index_map_file
        else:
            self.cached_tensors_file = None
            self.cached_index_map_file = None

        if self.cached_tensors_file:
            self.emb_cache = torch.tensor(np.load(str(self.cached_tensors_file)))#.to(device)
            self.cache_map = json.load(self.cached_index_map_file.open('r'))
        else:
            self.emb_cache = None
            self.cache_map = None

        if model_name:
            # Load up the model
            folder = TRAINED_MODELS_FOLDER / self.model_name
            ckpt = folder / 'best_checkpoint.pth'

            self.model = ContrastiveModel.load(ckpt, self.device)

        self.instantiated = True

    def encode(
            self,
            nodes: List[Node],
            normalize_values: bool = True,
            clear_cache: bool = False
    ) -> torch.Tensor:
        """
        Generates encodings/embeddings for a given set of Nodes.

        :param nodes: The nodes you want to encode
        :param normalize_values: Encode the normalized value of the node, not the raw value.
        :param clear_cache: If true, always fetch the embedding, do not use the stored embedding on the node obj.
        :return: A list of encodings per given Node
        """

        self.model.eval()

        output = []
        left_over = []
        left_over_idxs = []
        for idx, node in enumerate(nodes):
            if node.tmp.get('embedding') is not None and not clear_cache:
                output.append(node.tmp.get('embedding'))
            elif self.cache_map and node.normalized_value in self.cache_map:
                output.append(self.emb_cache[self.cache_map[node.normalized_value], :].to(self.device))
            else:
                left_over.append(node)
                left_over_idxs.append(idx)
                output.append(None)

        dataset = Dataset.from_dict({
            'values': [x.normalized_value if normalize_values else x.value for x in left_over],
            'output_idx': left_over_idxs,
            'idx': [idx for idx in range(len(left_over))]
        })
        dataset = dataset.map(lambda e: e, batched=True)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        with torch.no_grad():
            for batch in dataloader:
                batch_output = self.model.get_encodings(batch['values'])
                for oidx, nidx, bidx in zip(batch['output_idx'], batch['idx'], range(batch_output.shape[0])):
                    output[oidx] = batch_output[bidx].to(device)
                    left_over[nidx].tmp['embedding'] = batch_output[bidx]

        # return torch.stack(output).squeeze(1).to(self.device)
        return torch.stack(output).squeeze(1)


    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        return self.search_obj_type, {
            'model_name': self.model_name,
            'batch_size': self.batch_size,
            'device': self.device
        }

    def to(self, device: str) -> 'NodeEmbedder':
        if device == self.device:
            return self

        return NodeEmbedder(
            self.model_name,
            self.batch_size,
            device
        )
