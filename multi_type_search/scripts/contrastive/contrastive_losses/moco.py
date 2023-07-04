import torch
from torch import nn
from torch.nn import Module
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt

from multi_type_search.search.search_model.types.contrastive import contrastive_utils
from multi_type_search.scripts.contrastive.contrastive_losses import ContrastiveLoss


class MOCOLoss(ContrastiveLoss):

    def __init__(
            self,
            temperature: float = 0.5
    ):
        super().__init__()

        self.temperature = temperature

        self.closs = CrossEntropyLoss(ignore_index=-1)

    def forward(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
            positive_sample_matrix: torch.Tensor,
    ) -> torch.Tensor:

        scores = torch.mm(predictions, targets.t())
        targets = torch.arange(scores.size(0)).cuda()
        return self.closs(scores, targets)
