import torch
import torch.nn.functional as F
from foundation.kullback_leibler_div import kl_selected_tokens
import copy

def compute_rho_per_token(policy_logprobs: torch.Tensor, # (B, L, V)
                          old_policy_logprobs: torch.Tensor, # (B, L, V)
                          labels: torch.Tensor) -> torch.Tensor: # Labels: (B, L) 0 for invalid tokens 1 for valid tokens
    # Also note (by tianhao.yao), here B usually is G
    policy_logp = policy_logprobs.gather(
        dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1) # (B, L)

    old_logp = old_policy_logprobs.gather(
        dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1) # (B, L)

    return torch.exp(policy_logp - old_logp) # (B, L) # Policy Ratios

def compute_adv(rewards: torch.Tensor, # (G,)
                eps: float = 1e-8) -> torch.Tensor:
    adv = (rewards - rewards.mean()) / (rewards.std() + eps) # (G,)
    return adv.unsqueeze(1) # (G, 1)                                

def compute_grpo_loss(
    policy_logits: torch.Tensor, # (G, L, V) one group => B==G
    old_policy_logits: torch.Tensor, # (G, L, V)
    reference_logits: torch.Tensor, # (G, L, V) ref model logits
    labels: torch.Tensor, # (G, L)
    completion_mask: torch.Tensor, # (G, T) 0 or 1
    rewards: torch.Tensor, # (G,)
    clip_eps: float = 0.2,
    kl_coef: float = 0.01,
    eps: float = 1e-8,
) -> torch.Tensor:
    # GRPO loss for a SINGLE group (same prompt, G completions).
    # Includes KL penalty term (approx tokenwise: logp - logpref) averaged over completion tokens.

    adv_tok = compute_adv(rewards, eps=eps)  # (G, 1)

    policy_logprobs = F.log_softmax(policy_logits, dim=-1)
    with torch.no_grad():
        old_logprobs = F.log_softmax(old_policy_logits, dim=-1)
        ref_logprobs = F.log_softmax(reference_logits, dim=-1)
    rho = compute_rho_per_token(policy_logprobs, old_logprobs, labels)  # (G, L)

    unclipped = rho * adv_tok  # (G, L)
    clipped = torch.clamp(rho, 1.0 - clip_eps, 1.0 + clip_eps) * adv_tok # clipped_rho * adv_tok
    surrogate = torch.minimum(unclipped, clipped)  # (G, L)
    
    m = completion_mask.to(surrogate.dtype)  # (G, L)
    denom = m.sum().clamp(min=1.0) 
    policy_loss = (-surrogate * m).sum() / denom

    # Added by tianhao.yao: 避免长回复主导梯度，或可先在样本内部求平均，再对样本求平均
    # per_sample_loss = (-surrogate * m).sum(dim=1) / m.sum(dim=1) # (G,)
    # policy_loss = per_sample_loss.mean()

    # KL ≈ E_{a~pi_theta}[ log pi_theta(a|s) - log pi_ref(a|s) ]

    kl_per_token = kl_selected_tokens(
        policy_logprobs,
        ref_logprobs,
        labels=labels,
        method="schulman"
    ) # (G, L)

    kl_loss = (kl_per_token * m).sum() / denom
    total_loss = policy_loss + kl_coef * kl_loss
    
    return total_loss

def freeze(m: torch.nn.Module):
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)

# GRPO uses reverse KL and Distillation uses forward KL

# Every time before calling this function, we perform a new round of rollout
def grpo_train_fixed_ref(
    sft_policy: torch.nn.Module, 
    ref_policy: torch.nn.Module,
    dataloader, # One group per sample, for simplicity
    optimizer: torch.optim.Optimizer,
    get_reward, # reward fn: input_ids -> (G,)
    device: torch.device,
    # eos_token_id,
    num_epochs: int = 1,
    clip_eps: float = 0.2,
    kl_coef: float = 0.01,
    # old_sync_every: int = 1,
):
    # Assumptions: dataloader yields dict with already-aligned tensors:
    # input_ids: (G, T)
    # completion_mask: (G, T) aligned; completion=1 else 0
    # For simplicity, each batch == single group (same prompt, G completions)

    policy = sft_policy.to(device)
    policy.train()

    old_policy = copy.deepcopy(policy).to(device)
    freeze(old_policy)

    if ref_policy is None: # Else we assume reference policy is a deep copy and has been frozen
        ref_policy = copy.deepcopy(policy).to(device)
        freeze(ref_policy)

    opt_step = 0

    for epoch in range(num_epochs):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)[..., :-1] # (G, L)
            # labels = torch.roll(input_ids, shifts=-1, dims=1)
            # labels[:, -1] = eos_token_id
            labels = input_ids[..., 1:] # (G, L)
            completion_mask = batch["completion_mask"].to(device)[...:, 1:] # (G, L)

            policy_logits = policy(input_ids) # (G, L, V)

            with torch.no_grad():
                rewards = get_reward(input_ids).to(device) # (G,)
                old_logits = old_policy(input_ids) # (G, L, V)
                ref_logits = ref_policy(input_ids) # (G, L, V)

            loss = compute_grpo_loss(
                policy_logits=policy_logits,
                old_policy_logits=old_logits,
                reference_logits=ref_logits,
                labels=labels,
                completion_mask=completion_mask,
                rewards=rewards,
                clip_eps=clip_eps,
                kl_coef=kl_coef,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            opt_step += 1

            # if old_sync_every > 0 and (opt_step % old_sync_every == 0):
            #     old_policy.load_state_dict(policy.state_dict())
            #     freeze(old_policy)

            # I think old_policy will not change until we rollout again (in normal approaches)

    return policy
