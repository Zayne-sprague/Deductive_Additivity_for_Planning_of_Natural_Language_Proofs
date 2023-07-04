import torch
from torch import nn
from torch.nn import Module
from torch.nn import MSELoss

import matplotlib.pyplot as plt

from multi_type_search.search.search_model.types.contrastive import contrastive_utils
from multi_type_search.scripts.contrastive.contrastive_losses import ContrastiveLoss


class MSELoss(ContrastiveLoss):

    def __init__(
            self,
    ):
        super().__init__()

        self.mse_loss = MSELoss()

    def forward(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
            positive_sample_matrix: torch.Tensor,
    ) -> torch.Tensor:
        return self.mse_loss(predictions, targets)
