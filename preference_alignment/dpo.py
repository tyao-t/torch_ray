# $$L_{DPO}(\pi_{\theta}; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim D} 
# \left[ \log \sigma \left( \beta \log \frac{\pi_{\theta}(y_w | x)}{\pi_{ref}(y_w | x)} 
# - \beta \log \frac{\pi_{\theta}(y_l | x)}{\pi_{ref}(y_l | x)} \right) \right]$$

# $$L_{DPO} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} 
# [ \log \sigma ( r(x, y_w) - r(x, y_l) ) $$

import torch.nn.functional as F
import torch
from text_and_data.preference_dpo_dataset import PreferenceDatasetDPO, custom_collate_fn_dpo, eos_token_id
from torch.utils.data import DataLoader
from inference.generate import generate_text_stream_concat_flex
from transformers import AutoTokenizer
import copy

def compute_logprobs(
    logits, # (B, L+1, V)
    labels, # (B, L+1)
    selection_mask=None # (B, L+1)
):
    # labels = torch.concat(labels[:, 1:], torch.full_like(labels[:, :1], eos_token_id))
    # labels = torch.concat([labels[:, 1:], torch.full_like(labels[:, :1], eos_token_id)], dim=-1)

    # labels = torch.roll(labels, shifts=-1, dims=1)
    # labels[:, -1] = eos_token_id
    
    labels = labels[:, 1:].clone() # (B, L)
    logits = logits[:, :-1, :] # (B, L, V)

    log_probs = F.log_softmax(logits, dim=-1) # (B, L, V)

    selected_log_probs = torch.gather(
        input=log_probs, # (B, L, V)
        dim=-1,
        index=labels.unsqueeze(-1) # (B, L, 1)
    ).squeeze(-1) # (B, L)

    if selection_mask is None:
        return selected_log_probs.mean(-1) # (B,)

    # mask = selection_mask #.clone()
    mask = selection_mask[:, 1:] # (B, L)
    selected_log_probs = selected_log_probs * mask # (B, L)
    avg_log_prob = selected_log_probs.sum(-1) / mask.sum(-1) # (B,)

    return avg_log_prob # (B,)

def compute_dpo_loss(
      model_chosen_logprobs, # (B,)
      model_rejected_logprobs, # (B,)
      reference_chosen_logprobs, # (B,)
      reference_rejected_logprobs, # (B,)
      beta=0.1, # Temperature parameter (normally 0.1-0.5) controlling divergence from reference model.
    ):

    model_logratios = model_chosen_logprobs - model_rejected_logprobs # (B,)
    reference_logratios = reference_chosen_logprobs - reference_rejected_logprobs # (B,)
    logits = beta * (model_logratios - reference_logratios) # (B,)

    losses = -torch.nn.functional.logsigmoid(logits) # (B,)

    chosen_rewards = (model_chosen_logprobs - reference_chosen_logprobs).detach() # (B,)
    rejected_rewards = (model_rejected_logprobs - reference_rejected_logprobs).detach() # (B,)
    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean() # (1,)

def compute_dpo_loss_batch(batch, policy, ref_policy, beta):
    policy_chosen_log_probas = compute_logprobs(
        logits=policy(batch["chosen"]), # (B, L+1, V)
        labels=batch["chosen"], # (B, L+1)
        selection_mask=batch["chosen_mask"] # (B, L+1)
    ) # (B,)
    policy_rejected_log_probas = compute_logprobs(
        logits=policy(batch["rejected"]), # (B, L+1, V)
        labels=batch["rejected"], # (B, L+1)
        selection_mask=batch["rejected_mask"] # (B, L+1)
    ) # (B,)
    
    with torch.no_grad():
        ref_chosen_log_probas = compute_logprobs(
            logits=ref_policy(batch["chosen"]), # (B, L+1, V)
            labels=batch["chosen"], # (B, L+1)
            selection_mask=batch["chosen_mask"] # (B, L+1)
        ) # (B,)
        ref_rejected_log_probas = compute_logprobs(
            logits=ref_policy(batch["rejected"]), # (B, L+1, V)
            labels=batch["rejected"], # (B, L+1)
            selection_mask=batch["rejected_mask"] # (B, L+1)
        ) # (B,)

    loss, chosen_rewards, rejected_rewards = compute_dpo_loss(
        model_chosen_logprobs=policy_chosen_log_probas, # (B,)
        model_rejected_logprobs=policy_rejected_log_probas, # (B,)
        reference_chosen_logprobs=ref_chosen_log_probas, # (B,)
        reference_rejected_logprobs=ref_rejected_log_probas, # (B,)
        beta=beta
    )
    return loss, chosen_rewards, rejected_rewards # (1,)


train_data = {'instruction': '...', 'input': '...', 'chosen': '...', 'rejected': '...'}
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B-Instruct", trust_remote_code=True)

train_loader = DataLoader(
    PreferenceDatasetDPO(train_data, tokenizer),
    batch_size=8,
    collate_fn=custom_collate_fn_dpo,
    shuffle=True
)

def compute_dpo_loss_loader(data_loader, policy, ref_policy, beta, num_batches=None):
    # Apply compute_dpo_loss_batch to a whole data loader
    total_loss, total_chosen_rewards, total_rejected_rewards = 0., 0., 0.
    if len(data_loader) == 0:
        return float("nan")

    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, batch in enumerate(data_loader):
        if i < num_batches:
            loss, chosen_rewards, rejected_rewards = compute_dpo_loss_batch(
                batch=batch,
                policy=policy,
                ref_policy=ref_policy,
                beta=beta
            )
            total_loss += loss.item()
            total_chosen_rewards += chosen_rewards.item()
            total_rejected_rewards += rejected_rewards.item()
        else:
            break

    total_loss /= num_batches
    total_chosen_rewards /= num_batches
    total_rejected_rewards /= num_batches
    return total_loss, total_chosen_rewards, total_rejected_rewards

def evaluate_dpo_loss_loader(policy, ref_policy, train_loader, val_loader, beta, eval_iter):
    # Compute the DPO loss for the training and validation dataset
    policy.eval()
    with torch.no_grad():
        train_loss, train_chosen_rewards, train_rejected_rewards = compute_dpo_loss_loader(
            data_loader=train_loader,
            policy=policy,
            ref_policy=ref_policy,
            beta=beta,
            num_batches=eval_iter
        )

        val_loss, val_chosen_rewards, val_rejected_rewards = compute_dpo_loss_loader(
            data_loader=val_loader,
            policy=policy,
            ref_policy=ref_policy,
            beta=beta,
            num_batches=eval_iter
        )

    res = {
        "train_loss": train_loss,
        "train_chosen_reward": train_chosen_rewards,
        "train_rejected_reward": train_rejected_rewards,
        "val_loss": val_loss,
        "val_chosen_reward": val_chosen_rewards,
        "val_rejected_reward": val_rejected_rewards
    }

    policy.train()
    return res

def freeze(m: torch.nn.Module):
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)

def train_model_dpo_simple(
    policy, ref_policy, train_loader, val_loader,
    optimizer, num_epochs, beta, device,
    eval_freq, eval_iter, start_context, tokenizer
):
    if ref_policy is None: # Else we assume reference policy is a deep copy and has been frozen
        ref_policy = copy.deepcopy(policy).to(device)
        freeze(ref_policy)

    tracking = {
        "train_losses": [],
        "train_chosen_rewards": [],
        "train_rejected_rewards": [],
        "val_losses": [],
        "val_chosen_rewards": [],
        "val_rejected_rewards": [],
        "tokens_seen": []
    }
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        policy.train()

        for batch in train_loader:

            optimizer.zero_grad()

            loss, chosen_rewards, rejected_rewards = compute_dpo_loss_batch(
                batch=batch,
                policy=policy,
                ref_policy=ref_policy,
                beta=beta
            )

            loss.backward()
            optimizer.step()

            tokens_seen += batch["chosen"].numel() # (B*L,)
            global_step += 1

            if global_step % eval_freq == 0:
                res = compute_dpo_loss_loader(
                    policy=policy,
                    ref_policy=ref_policy,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    beta=beta,
                    eval_iter=eval_iter
                )
                tracking["train_losses"].append(res["train_loss"])
                tracking["train_chosen_rewards"].append(res["train_chosen_reward"])
                tracking["train_rejected_rewards"].append(res["train_rejected_reward"])
                tracking["val_losses"].append(res["val_loss"])
                tracking["val_chosen_rewards"].append(res["val_chosen_reward"])
                tracking["val_rejected_rewards"].append(res["val_rejected_reward"])
                tracking["tokens_seen"].append(tokens_seen)
                train_reward_margin = res["train_chosen_reward"] - res["train_rejected_reward"]
                val_reward_margin = res["val_chosen_reward"] - res["val_rejected_reward"]

                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {res['train_loss']:.3f}, Val loss {res['val_loss']:.3f}, "
                    f"Train reward margins {train_reward_margin:.3f}, "
                    f"Val reward margins {val_reward_margin:.3f}"
                )

        generate_text_stream_concat_flex(
            model=policy,
            tokenizer=tokenizer,
            device=loss.device,
            prompt=start_context,
            max_new_tokens=50
        )

    return tracking
