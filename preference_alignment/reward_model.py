import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from foundation.transformer_blocks import Qwen3TransformerBlock
from foundation.operators.normalizations import RMSNorm
import torch.nn.functional as F
class Qwen3RewardModel(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(
            cfg["vocab_size"], 
            cfg["emb_dim"], 
            dtype=cfg["dtype"]
        )
        
        self.transformer_blocks = nn.ModuleList([
            Qwen3TransformerBlock(cfg) for _ in range(cfg["n_layers"])
        ])
        
        self.final_norm = RMSNorm(cfg["emb_dim"])

        self.reward_head = nn.Linear(
            cfg["emb_dim"], 
            1, 
            bias=True, 
            dtype=cfg["dtype"]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        offset: int = 0,
        cache: Optional[Any] = None,
        mask: str = "causal",
    ) -> torch.Tensor:
        
        x = self.tok_emb(input_ids) # (B, L, D_in)

        for block in self.transformer_blocks:
            x = block(
                x, 
                offset=offset, 
                cache=cache, 
                mask=mask,
                exact=self.cfg.get("exact", False),
            )

        h = self.final_norm(x).to(self.cfg["dtype"]) # (B, L, D_in)
        return self.reward_head(h) # (B, L, 1)

def get_last_answer_token_reward(
    all_rewards: torch.Tensor, # (B, L, 1)
    mask: torch.Tensor # (B, L, 1)
) -> torch.Tensor: # （B, )
    B, L, _ = all_rewards.shape
    
    is_answer = (mask == 0).long()
    positions = torch.arange(L, device=all_rewards.device).unsqueeze(0)
    answer_indices = positions * is_answer
    last_idx = answer_indices.max(dim=1).view(B, 1, 1)
    
    final_scores = torch.gather(input=all_rewards,
        dim=1, 
        index=last_idx
    ).unsqueeze(0).unsqueeze(0)
    
    return final_scores

def preference_margin_loss(
    chosen_scores: torch.Tensor, # (B, 1)
    rejected_scores: torch.Tensor, # (B, 1)
    margin: torch.Tensor | float = 0.0 # (B, 1)
) -> torch.Tensor:
    # Preference Margin Loss in Llama 2.
    # L(theta) = -log(sigmoid(r_chosen - r_rejected - margin))
    # Idea: Want r_chosen - r_rejected > margin for model in order to reduce loss
    logits = chosen_scores - rejected_scores - margin
    loss = -F.logsigmoid(logits)
    return loss.mean()

def train_step(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    batch: dict,
    use_margin: bool = True
) -> dict:
    model.train()
    optimizer.zero_grad()
    
    c_ids = batch['chosen_input_ids'] # (B, L)
    c_mask = batch['chosen_mask'].unsqueeze(-1) # 0=Answer, 1=Others, including padding (B, L, 1)
    r_ids = batch['rejected_input_ids'] # (B, L)
    r_mask = batch['rejected_mask'].unsqueeze(-1) # 0=Answer, 1=Others, including padding (B, L, 1)
    
    c_full_rewards = model(c_ids, offset=0, kv_cache=None, mask="causal") # (B, L, 1)
    c_final_score = get_last_answer_token_reward(c_full_rewards, c_mask) # (B, )
    
    r_full_rewards = model(r_ids, offset=0, kv_cache=None, mask="causal")  # (B, L, 1)
    r_final_score = get_last_answer_token_reward(r_full_rewards, r_mask) # (B, )
    
    # Llama 2 uses a margin term m(y_c, y_r) based on rating difference
    margin = batch.get('margin', 0.0) if use_margin else 0.0
    
    loss = preference_margin_loss(c_final_score.unsqueeze(-1), r_final_score.unsqueeze(-1), margin).mean()
    
    loss.backward()
    optimizer.step()
    
    return {
        "loss": loss.item(),
        # "chosen_reward_mean": c_final_score.mean().item(),
        # "rejected_reward_mean": r_final_score.mean().item()
    }
