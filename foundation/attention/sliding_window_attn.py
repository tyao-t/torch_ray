import torch
import torch.nn as nn

class SlidingWindowCausalMHA(nn.Module):
    def __init__(self, d_in: int, d_out: int, num_heads: int, window_size: int, bias: bool = False):
        super().__init__()
        assert d_out % num_heads == 0
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.window_size = int(window_size)

        self.W_query = nn.Linear(d_in, d_out, bias=bias)
        self.W_key = nn.Linear(d_in, d_out, bias=bias)
        self.W_value = nn.Linear(d_in, d_out, bias=bias)
        self.out_proj = nn.Linear(d_out, d_in, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape # (B, L, d_in)
        
        q = self.W_query(x) # (B, L, d_out)
        k = self.W_key(x) # (B, L, d_out)
        v = self.W_value(x) # (B, L, d_out)

        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, L, D_h)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, L, D_h)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, L, D_h)

        B, H, L, D_h = q.shape # (B, H, L, D_h)
        W = min(self.window_size, L)
        scale = D_h ** -0.5
        neg_large = -torch.inf

        offsets = torch.arange(W, device=x.device, dtype=torch.long) - (W - 1) # (W)
        base = torch.arange(L, device=x.device, dtype=torch.long).unsqueeze(1) # (L, 1)
        idx = base + offsets.unsqueeze(0) # (L, W)

        invalid = idx < 0 # (L, W)
        idx = idx.clamp_min(0) # (L, W)

        idx_exp = idx.view(1, 1, L, W, 1).expand(B, H, L, W, D_h) # (B, H, L, W, D_h)

        k_expand = k.unsqueeze(3).expand(B, H, L, W, D_h) # (B, H, L, W, D_h)
        v_expand = v.unsqueeze(3).expand(B, H, L, W, D_h) # (B, H, L, W, D_h)

        k_win = torch.gather(k_expand, dim=2, index=idx_exp) # (B, H, L, W, D_h)
        v_win = torch.gather(v_expand, dim=2, index=idx_exp) # (B, H, L, W, D_h)

        scores = torch.matmul(q.unsqueeze(3), k_win.transpose(-1, -2)).squeeze(3) # (B, H, L, W)
        scores = scores * scale # (B, H, L, W)
        scores = scores.masked_fill(invalid.view(1, 1, L, W), neg_large) # (B, H, L, W)
        attn = torch.softmax(scores, dim=-1) # (B, H, L, W)

        ctx = torch.matmul(attn.unsqueeze(3), v_win).squeeze(3) # (B, H, L, D_h)

        ctx = ctx.transpose(1, 2).contiguous().view(B, L, self.d_out) # (B, L, d_out)
        return self.out_proj(ctx) # (B, L, d_in)
