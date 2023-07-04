import torch
from torch import nn
from torch.nn import Module
from pytorch_metric_learning.losses import NTXentLoss
from pytorch_metric_learning import losses as pml
from pytorch_metric_learning.distances import CosineSimilarity
import matplotlib.pyplot as plt


class ContrastiveLoss(nn.Module):

    def forward(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
            positive_sample_matrix: torch.Tensor,
    ):
        raise NotImplementedError('Implement forward passes for loss fns')
