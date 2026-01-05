import torch
import torch.nn.functional as F
from foundation.kullback_leibler_div import kl_selected_tokens
import copy

from grpo import compute_rho_per_token

def compute_ppo_clipped_policy_loss(
    policy_logprobs, # (B, L, V)
    old_logprobs, # (B, L, V)
    advantages, # (B, L)
    actions, # (B, L)
    mask, # (B, L)
    epsilon=0.2
):
    # rho = compute_rho_per_token(policy_logprobs, old_logprobs, actions) # (B, L)
    actions_unsqueezed = actions.unsqueeze(-1) # (B, L, 1)
    policy_logprobs = policy_logprobs.gather(-1, actions_unsqueezed).squeeze(-1) # (B, L)
    old_log_probs = old_logprobs.gather(-1, actions_unsqueezed).squeeze(-1) # (B, L)
    rho = torch.exp(policy_logprobs - old_log_probs) # (B, L)

    surr1 = rho * advantages # (B, L)
    surr2 = torch.clamp(rho, 1.0 - epsilon, 1.0 + epsilon) * advantages # (B, L)
    surrogate = torch.min(surr1, surr2) # (B, L)

    m = mask.to(surrogate.dtype) # (B, L)
    policy_loss = (-surrogate * m).sum() / m.sum().clamp(min=1.0)

    # Added by tianhao.yao： 避免长回复主导梯度，或可先在样本内部求平均，再对样本求平均
    # per_sample_loss = (-surrogate * m).sum(dim=1) / m.sum(dim=1) # (B,)
    # policy_loss = per_sample_loss.mean()

    return policy_loss

def compute_entropy_bonus(policy_logprobs: torch.Tensor, mask): # (B, L, V) and (B, L)
    # log_probs = F.log_softmax(policy_logits, dim=-1) # (B, L, V)
    probs = torch.exp(policy_logprobs) # (B, L, V)
    p_log_p = probs * policy_logprobs # (B, L, V)
    entropy_per_token = -torch.sum(p_log_p, dim=-1) # (B, L)
    m = mask.to(entropy_per_token.dtype) # (B, L)
    bonus = (entropy_per_token * m).sum() / m.sum().clamp(min=1.0)
    return bonus

def compute_value_targets(
    advantages: torch.Tensor, # (B, L)
    old_values: torch.Tensor # (B, L)
) -> torch.Tensor: # (B, L)
    targets = advantages + old_values

    return targets.detach()

def compute_value_loss(
    v_pred, # (B, L)
    v_target, # (B, L)
    mask,
    clip_val_loss=False,
    v_old=None, # (B, L)
    clip_range_vf=0.2
):
    if not clip_val_loss or v_old is None:
        loss_vf = (v_pred - v_target).pow(2)
    else:
        loss_vf_unclipped = (v_pred - v_target).pow(2)

        v_pred_clipped = v_old + torch.clamp(
            v_pred - v_old,
            -clip_range_vf,
            clip_range_vf
        )
        loss_vf_clipped = (v_pred_clipped - v_target).pow(2)

        loss_vf = torch.max(loss_vf_unclipped, loss_vf_clipped)

    loss_vf = loss_vf * 0.5
    m = mask.to(loss_vf.dtype) # (B, L)
    return (loss_vf * m).sum() / m.sum().clamp(min=1.0) # (B, L)

def compute_ppo_rewards(
    policy_logprobs: torch.Tensor, # (B, L, V)
    ref_logprobs: torch.Tensor, # (B, L, V)
    reward_scores: torch.Tensor, # (B,)
    actions: torch.Tensor, # (B, L)
    mask: torch.Tensor, # (B, L) 0/1
    kl_beta: float = 0.1
) -> torch.Tensor: # (B, L)
    # action_indices = actions.unsqueeze(-1) # (B, L, 1)
    # policy_lp_labels = policy_logprobs.gather(dim=-1, index=action_indices).squeeze(-1) # (B, L)
    # ref_lp_labels = ref_logprobs.gather(dim=-1, index=action_indices).squeeze(-1) # (B, L)

    kl_div = kl_selected_tokens(policy_logprobs, ref_logprobs, actions) # (B, L)
    rewards = -kl_beta * kl_div # (B, L)

    m = mask.to(rewards.dtype)
    rewards = rewards * m

    indices = torch.arange(m.shape[1], device=m.device) # (B, L)
    last_idx = (m * indices).argmax(dim=-1) # (B,)

    b_idx = torch.arange(rewards.shape[0], device=rewards.device)
    rewards[b_idx, last_idx] += reward_scores.to(rewards.dtype)

    return rewards # (B, L)

def compute_deltas(
    rewards: torch.Tensor, # r_t (B, L)
    mask: torch.Tensor, # (B, L)
    values: torch.Tensor, # V_old(t) (B, L)
    gamma: float = 0.99
) -> torch.Tensor: # (B, L)
    # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    # next_values is basically values shifted to the left by 1 position, and we pad zero to the very right
    m = mask.to(rewards.dtype)

    # shift V_{t+1} and if next token is invalid, treat V_{t+1} as 0
    next_values = torch.cat([values[:, 1:], torch.zeros_like(values[:, :1])], dim=1)
    next_m = torch.cat([m[:, 1:], torch.zeros_like(m[:, :1])], dim=1)
    # next_values = torch.roll(values, shifts=-1, dims=1)
    # next_values[:, -1] = 0
    # next_m = torch.zeros_like(m)
    # next_m[:, :-1] = m[:, 1:]
    next_values = next_values * next_m

    deltas = (rewards + gamma * next_values - values) * m
    return deltas

def compute_gae_advantages(
    deltas: torch.Tensor, # (B, L)
    mask: torch.Tensor, # (B, L), 0 = invalid, 1 = valid
    gamma: float = 0.99,
    lam: float = 0.95
) -> torch.Tensor: # (B, L)
    # A_t = (delta_t + gamma * lam * A_{t+1}) * mask_t

    B, L = deltas.shape
    advantages = torch.zeros_like(deltas) # (B, L)

    m = mask.to(deltas.dtype) # (B, L)
    next_advantage = torch.zeros(B, device=deltas.device, dtype=deltas.dtype) # (B,)

    for t in reversed(range(L)):
        current_advantage = (deltas[:, t] + (gamma * lam) * next_advantage) * m[:, t]
        # 反正这些是放在Torch.no_grad()下计算的，就暂先不detach了
        advantages[:, t] = current_advantage
        next_advantage = current_advantage

    return advantages

def freeze(m: torch.nn.Module):
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)

# Every time before calling this function, we perform a new round of rollout
def train_ppo_fixed_ref(
    policy_model: torch.nn.Module,
    value_model: torch.nn.Module,
    ref_policy: torch.nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    get_reward,
    # eos_token_id,
    device: torch.device,
    ppo_epochs: int = 4,
    clip_epsilon: float = 0.2,
    gamma: float = 0.99,
    lam: float = 0.95,
    kl_coef: float = 0.1,
    value_loss_coef: float = 0.5,
    entropy_coef: float = 0.01,
):
    policy = policy_model.to(device)
    policy.train()

    value_model = value_model.to(device)
    value_model.train()

    if ref_policy is None: # Else we assume reference policy is a deep copy and has been frozen
        ref_policy = copy.deepcopy(policy).to(device)
        freeze(ref_policy)

    # Snapshot old models once for this rollout-update
    old_policy = copy.deepcopy(policy_model).to(device)
    freeze(old_policy)

    old_value_model = copy.deepcopy(value_model).to(device)
    freeze(old_value_model)

    for epoch in range(ppo_epochs):
        for batch in dataloader:
            full_input_ids = batch["input_ids"].to(device) # (B, L)
            input_ids = full_input_ids[..., :-1] # (B, L)
            labels = full_input_ids[..., 1:] # (B, L)
            mask = batch["completion_mask"].to(device)[..., 1:] # (B, L), 0/1

            with torch.no_grad():
                reward_scores = get_reward(full_input_ids).to(device) # (B,)

                old_values = old_value_model(input_ids).squeeze(-1) # (B, L)

                old_logprobs = torch.log_softmax(old_policy(input_ids), dim=-1) # (B, L, V)
                ref_logprobs = torch.log_softmax(ref_policy(input_ids), dim=-1) # (B, L, V)

                rewards = compute_ppo_rewards(
                    policy_logprobs=old_logprobs,
                    ref_logprobs=ref_logprobs,
                    reward_scores=reward_scores,
                    actions=labels,
                    mask=mask,
                    kl_beta=kl_coef
                ) # (B, L)

                deltas = compute_deltas(
                    rewards=rewards,
                    values=old_values,
                    mask=mask,
                    gamma=gamma
                ) # (B, L)

                advantages = compute_gae_advantages(
                    deltas=deltas,
                    mask=mask,
                    gamma=gamma,
                    lam=lam
                ) # (B, L)

                returns = compute_value_targets(advantages, old_values) # (B, L)

                # Normalize一下Advantage，可选
                m = mask.to(advantages.dtype) # (B, L)
                valid = m > 0
                valid_advs = advantages[valid]
                advantages = (advantages - valid_advs.mean()) / valid_advs.std(unbiased=False).clamp(min=1e-8)
                advantages = advantages * m

                old_logprobs = old_logprobs.detach()
                old_values = old_values.detach()
                advantages = advantages.detach()
                returns = returns.detach()

            optimizer.zero_grad()

            policy_logprobs = torch.log_softmax(policy(input_ids), dim=-1) # (B, L, V)
            new_values = value_model(input_ids).squeeze(-1) # (B, L)

            loss_policy = compute_ppo_clipped_policy_loss(
                policy_logprobs=policy_logprobs,
                old_logprobs=old_logprobs,
                advantages=advantages,
                actions=labels,
                mask=mask,
                epsilon=clip_epsilon
            )

            loss_value = compute_value_loss(
                v_pred=new_values,
                v_target=returns,
                mask=mask,
                clip_val_loss=True,
                v_old=old_values,
                clip_range_vf=clip_epsilon
            )

            entropy = compute_entropy_bonus(policy_logprobs, mask)

            total_loss = loss_policy + (value_loss_coef * loss_value) - (entropy_coef * entropy)

            total_loss.backward()
            optimizer.step()

    return policy_model
