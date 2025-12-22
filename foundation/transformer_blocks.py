import torch.nn as nn
from foundation.feedforward.feedforward import Qwen23FeedForward #, GPT2FeedForward
from foundation.feedforward.mixture_of_experts import MoEFeedForward
from foundation.operators.normalizations import RMSNorm #, LayerNorm
from foundation.attention.grouped_query_attn import GroupedQueryAttention

class Qwen3TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gqa_attn = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out_qk=cfg["emb_dim"],
            d_out_v=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"],
            qkv_bias=False,
            cfg=cfg
        )
        self.ff = MoEFeedForward(cfg) if cfg["num_experts"] > 0 else Qwen23FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, cos, sin, *, offset=0, cache=None, mask="causal"):
        shortcut = x
        x = self.norm1(x)
        x = self.gqa_attn(x, cos, sin, offset = offset, cache=cache, mask=mask, cache=cache)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut

        return x

# class GPT2TransformerBlock(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.ln_1 = nn.LayerNorm(cfg["emb_dim"], bias=True)
#         self.ln_2 = nn.LayerNorm(cfg["emb_dim"], bias=True)
#         self.attn = MultiHeadAttention(d_in=cfg["emb_dim"], d_out=cfg["emb_dim"], \
#             num_heads=cfg["n_heads"], dropout=cfg["drop_rate"], qkv_bias=cfg["bias"])
#         self.ff = GPTFeedForward(cfg["emb_dim"], dropout=cfg["drop_rate"], bias=cfg["bias"])

#     def forward(self, x):
#         x_normalized = self.ln_1(x)
#         x = x + self.attn(x_normalized, x_normalized, x_normalized)
#         x = x + self.ff(self.ln_2(x))
#         return x