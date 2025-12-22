import torch

# GPT_CONFIG_124M = {
#     "vocab_size": 50304,
#     "context_length": 1024,
#     "emb_dim": 768,
#     "n_heads": 12,
#     "n_layers": 12,
#     "drop_rate": 0.1,
#     "bias": False
# }

# class GPTModel(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         assert cfg.vocab_size is not None
#         assert cfg.block_size is not None
#         self.cfg = cfg

#         self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
#         self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
#         self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
#         self.trf_blocks = nn.Sequential(
#             *(GPT2TransformerBlock(cfg) for _ in range(cfg["n_layers"]) ) 
#         )
        
#         self.ln_final = nn.LayerNorm(cfg["emb_dim"])
#         self.out_head = nn.Linear(
#             cfg["emb_dim"], cfg["vocab_size"], bias=False
#         )

#         self.apply(self._init_weights)

#         for pn, p in self.named_parameters():
#             if pn.endswith('out_proj.weight'):
#                 torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * cfg["n_layer"]))

#         print(f"Number of parameters: {self.get_num_params()/1e6:.2f}M")

#     def forward(self, in_idx):
#         batch_size, num_tokens = in_idx.shape
#         assert num_tokens <= self.cfg["context_length"], \
#             f"Unable to forward {num_tokens} tokens, max context_length is {self.cfg["context_length"]}"

#         tok_embeds = self.tok_emb(in_idx) # (batch_size, num_tokens, emb_dim)
#         # (num_tokens, emb_dim)
#         pos_embeds = self.pos_emb(torch.arange(num_tokens, device=in_idx.device))
        
#         x = tok_embeds + pos_embeds  # (batch_size, num_tokens, emb_dim)
#         x = self.drop_emb(x)
#         x = self.trf_blocks(x)
#         x = self.ln_final(x)

#         logits = self.out_head(x)
#         return logits

#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
#             if module.bias is not None:
#                 torch.nn.init.zeros_(module.bias)
#         elif isinstance(module, nn.Embedding):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
#         # elif isinstance(module, nn.LayerNorm): 
#         #     if module.weight is not None: torch.nn.init.ones_(module.weight)
#         #     if module.bias is not None: torch.nn.init.zeros_(module.bias)
#         #  The above last elif has been done by default for nn.LayerNorm

# LLAMA32_CONFIG_1B = {
#     "vocab_size": 128_256,
#     "context_length": 131_072,
#     "emb_dim": 2048,
#     "n_heads": 32,
#     "n_layers": 16,
#     "hidden_dim": 8192,
#     "n_kv_groups": 8,
#     "rope_base": 500_000.0,
#     "dtype": torch.bfloat16,
#     "rope_freq": {
#         "factor": 32.0,
#         "low_freq_factor": 1.0,
#         "high_freq_factor": 4.0,
#         "original_context_length": 8192,
#     }
# }

# class Llama3Model(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()

#         self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

#         self.trf_blocks = nn.ModuleList(
#             [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
#         )

#         self.final_norm = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
#         self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

#         cos, sin = compute_rope_params(
#             head_dim=cfg["emb_dim"] // cfg["n_heads"],
#             theta_base=cfg["rope_base"],
#             context_length=cfg["context_length"],
#             freq_config=cfg["rope_freq"]
#         )
#         self.register_buffer("cos", cos, persistent=False)
#         self.register_buffer("sin", sin, persistent=False)
#         self.cfg = cfg

#     def forward(self, in_idx):
#         tok_embeds = self.tok_emb(in_idx)
#         x = tok_embeds

#         num_tokens = x.shape[1]
#         mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)

#         for block in self.trf_blocks:
#             x = block(x, mask, self.cos, self.sin)
#         x = self.final_norm(x)
#         logits = self.out_head(x.to(self.cfg["dtype"]))
#         return logits

# class TransformerBlock(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.att = GroupedQueryAttention(
#             d_in=cfg["emb_dim"],
#             d_out=cfg["emb_dim"],
#             num_heads=cfg["n_heads"],
#             num_kv_groups=cfg["n_kv_groups"],
#             dtype=cfg["dtype"]
#         )
#         self.ff = FeedForward(cfg)
#         self.norm1 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
#         self.norm2 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])

#     def forward(self, x, mask, cos, sin):
#         shortcut = x
#         x = self.norm1(x)
#         x = self.att(x, mask, cos, sin)
#         x = x + shortcut 

#         shortcut = x
#         x = self.norm2(x)
#         x = self.ff(x)
#         x = x + shortcut

#         return x