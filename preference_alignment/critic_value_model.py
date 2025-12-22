import torch
import torch.nn as nn
from foundation.transformer_blocks import Qwen3TransformerBlock
from foundation.operators.normalizations import RMSNorm

# Sometimes, this Critic model share some or all transformers with the reward model.
# For simplicty, we let them be completely separate models here
class Qwen3CriticModel(nn.Module):
    # Input: (B, L)
    # Output: (B, L, 1)
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.transformer_blocks = nn.ModuleList([
            Qwen3TransformerBlock(cfg) for _ in range(cfg["n_layers"])
        ])
        self.final_norm = RMSNorm(cfg["emb_dim"])

        self.value_head = nn.Linear(cfg["emb_dim"], 1, bias=True)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.tok_emb(input_ids)
        for block in self.transformer_blocks:
            x = block(x, **kwargs)
        h = self.final_norm(x)

        # Input: (B, L, D)
        # Output: (B, L, 1)
        values = self.value_head(h)
        
        return values
