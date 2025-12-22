import torch
import torch.nn as nn
from foundation.operators.rotary_pos_embeddings import compute_positional_params
from foundation.operators.normalizations import RMSNorm
from foundation.transformer_blocks import Qwen3TransformerBlock

class Qwen3Model(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.transformer_blocks = nn.ModuleList(
            [Qwen3TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        cos, sin = compute_positional_params(
            head_dim=cfg["head_dim"],
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, in_token_ids, *, offset=0, cache=None, mask="causal"):
        x = self.tok_emb(in_token_ids)
        for block in self.transformer_blocks:
            x = block(
                x, cos=self.cos, sin=self.sin, offset=offset, cache=cache, mask=mask,
                exact=self.cfg["exact"],
            )

        logits = self.out_head(self.final_norm(x).to(self.cfg["dtype"]))
        return logits
