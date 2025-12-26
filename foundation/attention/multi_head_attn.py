import torch.nn as nn
import torch
import functools

def causal_mask(query_len, kv_len, device=None):
    device = device if device is not None else 'cpu'
    return torch.triu(torch.ones((query_len, kv_len), dtype=torch.bool, device=device),
                      diagonal=1 + kv_len - query_len)
    i = torch.arange(query_len, device=device).unsqueeze(1) # (query_len, 1)
    j = torch.arange(kv_len, device=device).unsqueeze(0) # (1, kv_len)
    return (j - i) >= (1 + kv_len - query_len) # bool (query_len, kv_len)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.attn_dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        *b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(*b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(*b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(*b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(-2, -3)
        queries = queries.transpose(-2, -3)
        values = values.transpose(-2, -3)

        attn_scores = queries @ keys.transpose(-1, -2)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / torch.sqrt(keys.shape[-1]), dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(-2, -3)

        context_vec = context_vec.contiguous().view(*b, num_tokens, self.d_out) # or .reshape
        context_vec = self.out_proj(context_vec)

        return context_vec # There would also be a resid dropout here normally

class MHASeq2SeqCompatible(nn.Module):
    def __init__(self, d_in, d_out_qk, d_out_v, context_length, dropout, num_heads, bias=False):
        super().__init__()
        assert d_out_qk % num_heads == 0 and d_out_v % num_heads == 0
        
        self.d_out_qk = d_out_qk
        self.d_out_v = d_out_v
        self.num_heads = num_heads
        self.head_dim_qk = d_out_qk // num_heads
        self.head_dim_v = d_out_v // num_heads

        self.W_query = nn.Linear(d_in, d_out_qk, bias=bias)
        self.W_key = nn.Linear(d_in, d_out_qk, bias=bias)
        self.W_value = nn.Linear(d_in, d_out_v, bias=bias)
        self.out_proj = nn.Linear(d_out_v, d_in, bias=bias)
        
        self.attn_dropout = nn.Dropout(dropout)

        # self.register_buffer(
        #     "mask",
        #     torch.triu(torch.ones(context_length, context_length), diagonal=1)
        # )

    def forward(self, x_q, x_k=None, x_v=None, *, mask="causal"):
        if x_k is None: x_k = x_q
        if x_v is None: x_v = x_q

        batch_size, num_tokens_q, _ = x_q.shape
        _, num_tokens_k, _ = x_k.shape
        _, num_tokens_v, _ = x_v.shape
        assert num_tokens_k == num_tokens_v, "num_tokens_k and num_tokens_v must match"

        queries = self.W_query(x_q).view(batch_size, num_tokens_q, self.num_heads, self.head_dim_qk).transpose(-2, -3)
        keys    = self.W_key(x_k).view(batch_size, num_tokens_k, self.num_heads, self.head_dim_qk).transpose(-2, -3)
        values  = self.W_value(x_v).view(batch_size, num_tokens_v, self.num_heads, self.head_dim_v).transpose(-2, -3)

        attn_scores = queries @ keys.transpose(-1, -2)  # (b, num_heads, num_tokens_q, num_tokens_k)

        # if is_causal and num_tokens_q == num_tokens_k:
        #     mask_bool = self.mask.bool()[:num_tokens_q, :num_tokens_k]
        mask = causal_mask(num_tokens_q, num_tokens_k, device=x_q.device) if mask == "causal" else mask
        if mask is not None: attn_scores.masked_fill_(mask, -torch.inf)

        attn_weights = torch.softmax(attn_scores / torch.sqrt(self.head_dim_qk), dim=-1)
        attn_weights = self.attn_dropout(attn_weights) # Still # (b, num_heads, num_tokens_q, num_tokens_k)

        context_vec = attn_weights @ values  # (b, num_heads, num_tokens_q, head_dim_v)
        context_vec = context_vec.transpose(-2, -3).contiguous().view(batch_size, num_tokens_q, self.d_out_v)
        context_vec = self.out_proj(context_vec)

        return context_vec # Plus resid dropout

class MultiHeadAttentionOptimizedSDPA(nn.Module):
    def __init__(self, d_in, d_out, num_heads, dropout=0, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "d_out must be a multiple of num_heads"

        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        batch_size, num_tokens, embed_dim = x.shape

        # (batch_size, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (3, batch_size, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # 3 x (batch_size, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv.unbind(0)

        # (batch_size, num_heads, num_tokens, head_dim)
        attn = functools.partial(nn.functional.scaled_dot_product_attention, 
                queries, keys, values, dropout_p=self.dropout if self.training else 0)
        context_vec = attn(is_causal=True) if mask == "causal" else attn(attn_mask=mask)
 
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

        return self.resid_dropout(self.out_proj(context_vec))
