import torch
from torch import nn
from torch.nn import Module
from pytorch_metric_learning.losses import NTXentLoss
from pytorch_metric_learning import losses as pml
from pytorch_metric_learning.distances import CosineSimilarity
import matplotlib.pyplot as plt

from multi_type_search.search.search_model.types.contrastive import contrastive_utils
from multi_type_search.scripts.contrastive.contrastive_losses import ContrastiveLoss


class NTXENTLoss(ContrastiveLoss):

    def __init__(
            self,
            temperature: float = 0.5
    ):
        super().__init__()

        self.temperature = temperature

        self.ntxentloss = NTXentLoss(temperature=temperature)

    def forward(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
            positive_sample_matrix: torch.Tensor,
    ) -> torch.Tensor:

        predictions = torch.cat((targets, predictions), dim=0)


        indices = torch.arange(0, targets.size(0), device=targets.device)
        labels = torch.cat((indices, indices))

        return self.ntxentloss(predictions, labels)
