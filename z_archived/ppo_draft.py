# import torch

# # --- Example Usage ---

# # Mock Data parameters
# batch_size, num_tokens, vocab_size = 1, 10, 100

# # 1. Inputs (Policy)
# new_policy_logits = torch.randn(batch_size, num_tokens, vocab_size, requires_grad=True)
# old_policy_logits = torch.randn(batch_size, num_tokens, vocab_size) # Fixed
# fake_actions = torch.randint(0, vocab_size, (batch_size, num_tokens))
# fake_advantages = torch.randn(batch_size, num_tokens)

# # 2. Inputs (Value Function)
# # v_pred is the output of the CURRENT Critic model (This is what we are training)
# v_pred = torch.randn(batch_size, num_tokens, requires_grad=True) 

# # v_old is what the Critic predicted during Rollout (Fixed Data from buffer)
# # It is NEVER updated during this training loop.
# v_old = torch.randn(batch_size, num_tokens) 

# # v_target is pre-computed as (Advantage + v_old)
# v_target = fake_advantages + v_old 


# # 3. Compute Terms
# loss_policy = compute_ppo_clip_loss(
#     new_policy_logits, 
#     old_policy_logits, 
#     fake_advantages, 
#     fake_actions
# )

# loss_entropy = compute_entropy_bonus(new_policy_logits)

# loss_value = compute_value_loss(
#     v_pred, 
#     v_target, 
#     clip_val_loss=False # Set to True to use v_old for clipping
# )

# # 4. Final Total Loss
# # Coefficients are hyperparameters (standard values shown)
# coef_vf = 0.5   # Critic coefficient (usually 0.5 or 1.0)
# coef_ent = 0.01 # Entropy coefficient (encourages exploration)

# # Total = Policy Loss + (c_vf * Value Loss) - (c_ent * Entropy Bonus)
# total_loss = loss_policy + (coef_vf * loss_value) - (coef_ent * loss_entropy)

# print(f"Policy Loss (L^CLIP): {loss_policy.item():.4f}")
# print(f"Value Loss (L^VF):    {loss_value.item():.4f}")
# print(f"Entropy Bonus:        {loss_entropy.item():.4f}")
# print("-" * 30)
# print(f"Total Loss:           {total_loss.item():.4f}")

    


# # --- Test ---
# B, T, V = 2, 5, 10
# # Mock Data
# policy_dist = torch.randn(B, T, V).log_softmax(dim=-1)
# ref_dist    = torch.randn(B, T, V).log_softmax(dim=-1)
# actions     = torch.randint(0, V, (B, T)) # The tokens the model actually generated
# scores      = torch.tensor([[10.0], [5.0]]) # Reward model liked seq 1, disliked seq 2

# rewards = compute_ppo_rewards(policy_dist, ref_dist, scores, actions, beta=0.1)

# print("Rewards Shape:", rewards.shape) # [2, 5]
# print("Last token reward (Seq 1):", rewards[0, -1].item()) # Should be close to 10.0





# # --- Example Usage ---
# # Using the deltas from the previous step
# # deltas = [Batch, T]

# # Parameters
# gamma = 0.99
# lam = 0.95

# # Calculate
# advantages = compute_gae_advantages(deltas, gamma, lam)

# # Note: In PPO, it is standard to NORMALIZE advantages afterward
# # This stabilizes training significantly.
# # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

