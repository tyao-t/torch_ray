import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dim, *, eps=1e-5, bias=True):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        # var = x.var(dim=-1, keepdim=True, correction=0) # unbiased=False
        mean_square = torch.mean(torch.pow(x, 2), dim=-1, keepdim=True)
        norm_x = (x - mean) * torch.rsqrt((mean_square-torch.pow(mean, 2)) + self.eps)

        norm_x = norm_x * self.scale
        if self.shift is not None:
            norm_x = norm_x + self.shift
        return norm_x
    
class RMSNorm(nn.Module):
    def __init__(self, emb_dim, *, eps=1e-6, qwen_compliant=True, bias=False):
        super().__init__()
        self.eps = eps
        self.qwen_compliant = qwen_compliant
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype

        if self.qwen_compliant: x = x.to(torch.float32)

        mean_square = torch.mean(torch.pow(x, 2), dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(mean_square + self.eps)
        
        norm_x = norm_x * self.scale
        if self.shift is not None:
            norm_x = norm_x + self.shift
        return norm_x.to(input_dtype)
