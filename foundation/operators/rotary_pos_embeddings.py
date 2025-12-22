import torch
import torch.nn as nn

def compute_positional_params(head_dim, theta_base=10000, context_len=2048, dtype=torch.float32):
    assert head_dim % 2 == 0,  "head_dim must be even"
    half_dim = head_dim // 2

    inner = torch.arange(half_dim, dtype=dtype) / half_dim
    inv_freqs = torch.pow(theta_base, -inner)
    positions = torch.arange(context_len, dtype=dtype)
    angles = torch.outer(positions, inv_freqs)

    return torch.cos(angles), torch.sin(angles)

def apply_rotary_embedding(x, cos, sin, *, offset: int | list[int] | None = None, traditional=False):
    batch_size, num_heads, num_tokens, head_dim = x.shape
    assert num_tokens <= cos.shape[0]
    if offset is None:
        # (num_tokens, head_dim)
        cos = cos[:num_tokens, :]
        sin = sin[:num_tokens, :]
    elif isinstance(offset, int):
        # (num_tokens, head_dim)
        assert offset + num_tokens <= cos.shape[0]
        cos = cos[offset : offset + num_tokens, :]
        sin = sin[offset : offset + num_tokens, :]
    elif isinstance(offset, list) or isinstance(offset, torch.Tensor):
        if isinstance(offset, list):
            for o in offset: assert num_tokens + o <= cos.shape[0]
            offset = torch.tensor(offset, device=x.device)
        else:
            for o in offset.squeeze(-1): assert num_tokens + o.item() <= cos.shape[0]

        assert num_tokens + offset <= cos.shape[0]
        idx = torch.arange(num_tokens, device=x.device).unsqueeze(0) + offset.unsqueeze(1)
        
        # (batch_size, num_tokens, head_dim) 
        cos = cos[idx]
        sin = sin[idx]

    # (batch_size, 1, num_tokens, head_dim) or (1, 1, num_tokens, head_dim) but (num_tokens, head_dim) also works
    # to make sure it matches (batch, heads, tokens, dim)
    cos = cos.unsqueeze(1) if cos.ndim == 3 else cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(1) if sin.ndim == 3 else sin.unsqueeze(0).unsqueeze(0)

    half_dim = head_dim // 2

    if traditional:
        # [x1, x2, x1, x2]
        x_reshaped = x.view(batch_size, num_heads, num_tokens, half_dim, 2)
        x1 = x_reshaped[..., 0]
        x2 = x_reshaped[..., 1]
    else:
        # [x1, x1, x2, x2]
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]

    real = x1 * cos - x2 * sin
    imag = x1 * sin + x2 * cos

    if traditional:
        x_rotated = torch.stack([real, imag], dim=-1)
    else:
        x_rotated = torch.cat([real, imag], dim=-1)

    # r*(x1+x2*i)*(cos_t+sin_t*i), where r = 1
    # = x1cos_t+x1_sin_t*i+x2*cos_t*i+x2_sin_t*i*i
    # = x1cos_t-x2sin_t+(x1sin_t+x2cos_t)i

    # cos(t1)cos(t2)-sin(t1)sin(t2) + [cos(t1)sin(t2) + cos(t2)sin(t1)]i

    # a^2 + b^2 + 2ab = (a+b) ^ 2 = a(a+b) + b(a+b) = x^2 + y^2 

    # .contiguous().view()
    return x_rotated.reshape(batch_size, num_heads, num_tokens, head_dim).to(dtype=x.dtype)

class SinusoidalAbsolutePositionalEncoder(nn.Module): # 2017, in Attention is all you need paper (Added by tianhao.yao)
    def __init__(self, emb_dim, max_seq_len=2048, dropout=0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, emb_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-torch.log(10000.0) / emb_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x * torch.sqrt(self.emb_dim)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
