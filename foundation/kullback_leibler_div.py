import torch
import torch.nn.functional as F

def kl_selected_tokens(
    log_probs_policy: torch.Tensor, # (B, L, V) log πpolicy
    log_probs_ref: torch.Tensor, # (B, L, V) log πref
    labels: torch.Tensor, # (B, L, 1) token ids
    # mask: torch.Tensor, # (B, L, 1) 0 for invalid tokens and 1 for valid
    method: str = "schulman", # "schulman" or "simple"
) -> torch.Tensor: # (B, L)
    lp_pol = log_probs_policy.gather(dim=-1, index=labels).squeeze(-1)
    lp_ref = log_probs_ref.gather(dim=-1, index=labels).squeeze(-1)

    if method == "simple":
        kl_val = lp_pol - lp_ref
    elif method == "schulman":
        delta = lp_ref - lp_pol
        kl_val = torch.exp(delta) - delta - 1.0
    else:
        raise ValueError(f"Unknown method={method}")

    return kl_val # * mask.to(kl_val.dtype)

def kl_full_distribution(log_probs_policy: torch.Tensor,
                                log_probs_ref: torch.Tensor) -> torch.Tensor:
    # KL(π_policy || π_ref) = Σ_v π_policy(v) * (log π_policy(v) - log π_ref(v))
    probs_policy = log_probs_policy.exp()  # (B, L, V)
    return (probs_policy * (log_probs_policy - log_probs_ref)).sum(dim=-1)  # (B L)

def kl_full_distribution(log_probs_policy: torch.Tensor,
                               log_probs_ref: torch.Tensor) -> torch.Tensor:
    probs_policy = log_probs_policy.exp()

    # kl_div returns shape (B, L, V) when reduction='none'
    kl_per_vocab = F.kl_div(
        input=log_probs_ref, # log q
        target=probs_policy, # p
        reduction='none',
        log_target=False
    )
    return kl_per_vocab.sum(dim=-1)  # (B, L)
