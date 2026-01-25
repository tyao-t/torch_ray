import torch
import torch.nn as nn
from foundation.operators.fundamental_ops import swish, gelu

class Qwen23FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.up_proj = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"]*2, dtype=cfg["dtype"], bias=False)
        self.down_proj = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        candidate, gate_input = self.up_proj(x).chunk(2, dim=-1)
        gate = swish(gate_input)
        gated = candidate * gate
        return self.down_proj(gated)
    
class Qwen23FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.up_proj = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"]*2, dtype=cfg["dtype"], bias=False)
        self.down_proj = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)
        self.beta = nn.Parameter(torch.ones(cfg["hidden_dim"], dtype=cfg["dtype"]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        candidate, gate_input = self.up_proj(x).chunk(2, dim=-1)
        gate = gate_input * torch.nn.functional.sigmoid(self.beta * gate_input)
        gated = candidate * gate
        return self.down_proj(gated)
    
# class Qwen23FeedForward(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.w_up = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
#         self.w_gate = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
#         self.w_down = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

#     def forward(self, x):
#         candidate = self.w_up(x)
#         gate = swish(self.w_gate(x))
#         gated = gate * candidate
#         return self.w_down(gated)
    
class GPT2FeedForward(nn.Module):
    def __init__(self, emb_dim, *, dropout, bias):
        super().__init__()
        self.out_proj = nn.Linear(4 * emb_dim, emb_dim, bias=bias)
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim, bias=bias),
            nn.GELU(approximate="none"), # nn.GELU(approximate="tanh")
            self.out_proj,
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.layers(x)

FeedForward = GPT2FeedForward
FeedForward = Qwen23FeedForward
