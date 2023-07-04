import torch


def l2_simularity_metric(embedding_1: torch.Tensor, embedding_2: torch.Tensor) -> torch.Tensor:
    return ((embedding_1 - embedding_2) ** 2).mean(-1)


def cosine_similarity_metric(embedding_1: torch.Tensor, embedding_2: torch.Tensor, dim=-1) -> torch.Tensor:
    # return torch.dot(embedding_1, embedding_2) / (torch.norm(embedding_1) * torch.norm(embedding_2))
    # return (embedding_1 * embedding_2).sum(-1) / (torch.norm(embedding_1) * torch.norm(embedding_2))
    return torch.nn.functional.cosine_similarity(embedding_1, embedding_2, dim)

def spherical_norm(embedding: torch.Tensor) -> torch.Tensor:
    embedding = embedding - embedding.min()
    embedding = embedding / embedding.max()
    embedding = embedding * 2 - 1
    return embedding
