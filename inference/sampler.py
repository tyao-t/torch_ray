import torch 
import torch.nn.functional as F
import functools

greedy_sampler = functools.partial(torch.argmax, dim=-1, keepdim=False)

# HuggingFace normally applies 1. Temperature 2. Top-K 3. Top-P

def top_p_filter(probs, # (B, L, input_size) where input_size might be vocab_size or top_k
                 top_p):
    if top_p is None or top_p <= 0.0 or top_p >= 1.0:
        return probs

    sorted_probas, sorted_idx = torch.sort(probs, dim=-1, descending=True) # (B, L, input_size)
    cumprobas = torch.cumsum(sorted_probas, dim=-1) # (B, L, input_size)

    keep = cumprobas <= top_p # (B, L, input_size) bool
    keep[...,0] = True

    kept_sorted = torch.where(
        condition=keep, input=sorted_probas,
        other=torch.zeros_like(sorted_probas)
    )
    filtered = torch.zeros_like(probs).scatter(dim=-1, index=sorted_idx, src=kept_sorted)

    filtered = F.normalize(filtered, p=1, dim=-1) # (B, L, input_size)
    # denom = torch.sum(filtered, dim=-1).clamp_min(1e-12)
    return filtered # filtered / denom 

def create_sampler(temperature=0, top_p=1.0, top_k=None):
    def sample(logits: torch.Tensor): # (B, L, vocab_size)
        if temperature <= 0.0:
            return greedy_sampler(logits) # (B, L)

        logits = logits / temperature if temperature != 1.0 else logits

        if top_k is not None and top_k > 0:
            topk_logits, topk_pos = torch.topk(logits, k=top_k, dim=-1) # (B, L, top_k)
            probs = top_p_filter(torch.softmax(topk_logits, dim=-1), top_p) # (B, L, top_k)
            # min_val = top_logits[:, -1]
            # logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)
            # A relatively better implementation:
            probs = torch.zeros_like(probs).scatter(dim=-1, index=topk_pos, src=probs) # (B, L, vocab_size)
        else:
            probs = top_p_filter(torch.softmax(logits, dim=-1), top_p) # (B, L, vocab_size)

        return torch.multinomial(probs, num_samples=1, replacement=False).squeeze(-1) # (B, L)

    return sample
