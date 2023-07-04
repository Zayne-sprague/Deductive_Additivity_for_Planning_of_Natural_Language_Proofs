import sys

import torch
from torch import nn
from torch.nn import Module
from pytorch_metric_learning.losses import NTXentLoss, ContrastiveLoss
from pytorch_metric_learning import losses as pml
from pytorch_metric_learning.distances import CosineSimilarity
import matplotlib.pyplot as plt

from multi_type_search.search.search_model.types.contrastive.contrastive_utils import cosine_similarity_metric
from multi_type_search.scripts.contrastive.contrastive_losses import ContrastiveLoss


class CosineLoss(nn.Module):

    def forward(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
            positive_sample_matrix: torch.Tensor,
    ):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        positive_sample_matrix = positive_sample_matrix.to(device)

        batch_size = predictions.shape[0]

        negative_sample_matrix = (positive_sample_matrix == 0).float()

        # x = nn.functional.normalize(embeddings)
        # y = nn.functional.normalize(targets)

        similarity_matrix = nn.functional.cosine_similarity(
            predictions.unsqueeze(1),
            targets,
            dim=2
        )

        # data = similarity_matrix.to('cpu').detach().numpy()
        # plt.figure(figsize=(5, 5))
        # im = plt.imshow(data)
        # plt.colorbar(im)
        # plt.show()


        similarity_matrix = similarity_matrix.to(device)

        similarity_matrix /= 0.07

        max_val = torch.max(similarity_matrix).detach()
        similarity_matrix -= max_val

        positive_similarity_matrix = (similarity_matrix * positive_sample_matrix)
        positive_similarity_vector = positive_similarity_matrix.sum(-1)
        # positive_similarity_vector = torch.diag(similarity_matrix)

        numerator = torch.exp(positive_similarity_vector)

        negative_similarity_matrix = (similarity_matrix * negative_sample_matrix)
        negative_similarity_vector = negative_similarity_matrix.sum(-1)

        denominator = numerator + torch.exp(negative_similarity_vector)

        losses = -torch.log(numerator / denominator) + 1e-7
        loss = losses.mean()
        return loss

    def _forward(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
            positive_sample_matrix: torch.Tensor,
    ):

        S = cosine_similarity_metric(predictions, targets)

        nc = -1 #(1 / max(int(targets.size(0) * 0.9), 1))
        M = torch.full([targets.size(0), targets.size(0)], nc).cuda()
        # M.fill_diagonal_(-1)
        M[positive_sample_matrix == 1] = 1


        loss_matrix = (M - S) ** 2

        # data = loss_matrix.to('cpu').detach().numpy()
        # plt.figure(figsize=(5, 5))
        # im = plt.imshow(data)
        # plt.colorbar(im)
        # plt.show()

        loss = loss_matrix.mean()

        # sys.exit(0)
        return loss
