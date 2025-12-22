import torch

def l2_distance(a: torch.Tensor, b: torch.Tensor, from_scratch=False) -> torch.Tensor:
    return torch.sqrt(torch.sum((a - b)**2)) if from_scratch else torch.norm(a - b, p=2) 

def inner_product(a: torch.Tensor, b: torch.Tensor, from_scratch=False) -> torch.Tensor:
    return torch.sum(a * b) if from_scratch else torch.dot(a, b)

def cosine_similarity(a: torch.Tensor, b: torch.Tensor, from_scratch=False) -> torch.Tensor:
    if from_scratch:
        dot_product = inner_product(a, b, from_scratch=from_scratch)
        norm_a = torch.sqrt(torch.sum(a**2))
        norm_b = torch.sqrt(torch.sum(b**2))
        return dot_product / (norm_a * norm_b)

    return torch.nn.CosineSimilarity(dim=0)(a, b)
