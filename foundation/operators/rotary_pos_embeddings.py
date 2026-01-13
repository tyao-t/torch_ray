import torch
import torch.nn as nn

def compute_positional_params(head_dim, theta_base=10000, context_len=2048, dtype=torch.float32, freq_config=None):
    assert head_dim % 2 == 0,  "head_dim must be even"
    half_dim = head_dim // 2

    inner = torch.arange(half_dim, dtype=dtype) / half_dim
    inv_freqs = torch.pow(theta_base, -inner)
    positions = torch.arange(context_len, dtype=dtype)
    angles = torch.outer(positions, inv_freqs)

    if freq_config is not None:
        min_turns = freq_config["low_freq_factor"]
        max_turns = freq_config["high_freq_factor"]
        rotations = angles * freq_config["original_context_length"] / (2 * torch.pi)
        angles_new = torch.where(rotations < min_turns, angles / freq_config["factor"], angles)
        smooth_factor = (rotations - min_turns) / (max_turns - min_turns)
        angles_smoothed = (1 - smooth_factor) * (angles / freq_config["factor"]) + smooth_factor * angles
        is_medium_freq = (rotations >= min_turns) & (rotations <= max_turns)
        angles_new = torch.where(is_medium_freq, angles_smoothed, angles_new)
        angles = angles_new

    return torch.cos(angles), torch.sin(angles)

def apply_rotary_embedding(x, cos, sin, *, offset: int | list[int] | None = None, traditional=False):
    B, num_heads, L, head_dim = x.shape
    assert L <= cos.shape[0]
    if offset is None:
        # (L, head_dim)
        cos = cos[:L, :]
        sin = sin[:L, :]
    elif isinstance(offset, int):
        # (L, head_dim)
        assert offset + L <= cos.shape[0]
        cos = cos[offset : offset + L, :]
        sin = sin[offset : offset + L, :]
    elif isinstance(offset, list):
        offset = torch.tensor(offset, device=x.device)
        assert (offset + L).max() <= cos.shape[0]
        # offset: (B, 1) + arange: (1, L) = (B, L)
        idx = offset.unsqueeze(1) + torch.arange(L, device=x.device).unsqueeze(0)
        
        # (B, L, head_dim) 
        cos = cos[idx]
        sin = sin[idx]

    # (B, 1, L, head_dim) or (1, 1, L, head_dim) but (L, head_dim) also works
    # to make sure it matches (batch, heads, tokens, dim)
    cos = cos.unsqueeze(1) if cos.ndim == 3 else cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(1) if sin.ndim == 3 else sin.unsqueeze(0).unsqueeze(0)

    half_dim = head_dim // 2

    if traditional:
        # [x, y, x, y]
        x_reshaped = x.view(B, num_heads, L, half_dim, 2)
        real_cur = x_reshaped[..., 0]
        imag_cur = x_reshaped[..., 1]
    else:
        # [x, x, y, y]
        real_cur = x_reshaped[..., :half_dim]
        imag_cur = x_reshaped[..., half_dim:]

    real = real_cur * cos - imag_cur * sin
    imag = real_cur * sin + imag_cur * cos

    if traditional:
        x_rotated = torch.stack([real, imag], dim=-1)
    else:
        x_rotated = torch.cat([real, imag], dim=-1)

    # r*(x+y*i)*(cos_t+sin_t*i), where r = 1
    # = xcos_t+x_sin_t*i+y*cos_t*i+y_sin_t*i*i
    # = xcos_t-ysin_t+(xsin_t+ycos_t)i

    # cos(t1)cos(t2)-sin(t1)sin(t2) + [cos(t1)sin(t2) + cos(t2)sin(t1)]i

    # re^{ia} == r(cosa + isina)
    # re^{ia} * e^{ib} = e^{i(a+b)} = r[cos(a+b) + isin(a+b)] = r[cosacosb - sinasinb + cosasinbi + sinacosbi]
    # = (rcosa)cosb  + (rsina)*(-sinb) + i[(rcosa)sinb + (rsina)cosb] = xcosb - ysinb + i(xsinb + ycosb) = x' + y'
    
    # .contiguous().view()
    return x_rotated.reshape(B, num_heads, L, head_dim).to(dtype=x.dtype)

class SinusoidalAbsolutePositionalEncoder(nn.Module): # 2017, in Attention is all you need paper (Added by tianhao.yao)
    def __init__(self, emb_dim, max_seq_len=2048, dropout=0.1):
        super().__init__()
        assert emb_dim % 2 == 0, "emb_dim must be even"
        self.dropout = nn.Dropout(dropout)
        half_dim = emb_dim // 2
        inner = torch.arange(half_dim, dtype=torch.float32) / half_dim # (half_dim,)
        inv_freqs = torch.pow(10000, -inner) # (half_dim,)
        positions = torch.arange(max_seq_len, dtype=torch.float32) # (max_len,)
        
        angles = torch.outer(positions, inv_freqs) # (max_len, half_dim)
        
        sin_vals = torch.sin(angles) # (max_len, half_dim)
        cos_vals = torch.cos(angles) # (max_len, half_dim)
        
        pe = torch.stack([sin_vals, cos_vals], dim=-1).reshape(max_seq_len, emb_dim) # (max_len, 2)        
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        B, L, D = x.shape
        # x = x * torch.sqrt(self.emb_dim)
        x = x + self.pe[:L, :]
        return self.dropout(x)
