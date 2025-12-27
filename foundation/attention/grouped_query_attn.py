import torch
import torch.nn.functional as F
import torch.nn as nn
from foundation.operators.fundamental_ops import softmax
from foundation.operators.normalizations import RMSNorm
from foundation.operators.rotary_pos_embeddings import apply_rotary_embedding, compute_positional_params
from torch.nn.attention import sdpa_kernel, SDPBackend
from foundation.attention.multi_head_attn import causal_mask
from inference.kv_cache import KvCache

def scaled_dot_product_attention(queries, keys, values, *, scale=None, attn_mask=None, dropout_p=0, \
                                 enable_gqa=False, is_causal=False, **kwargs):
    assert attn_mask is None or (not is_causal), "Invalid when both attn_mask and is_causal are set"
    *leading_dims_q, num_heads, query_len, query_head_dim = queries.shape
    *leading_dims_k, num_groups, keys_len, keys_head_dim = keys.shape
    assert leading_dims_q == leading_dims_k

    if enable_gqa:
        assert num_heads % num_groups == 0
        assert num_groups == values.shape[-3]
        group_size = num_heads // num_groups
        if num_groups > 1:
            keys = keys.repeat_interleave(group_size, -3) # repeat also works (and has DRAM burst, memory coalescing), but why repeat_interleave  https://gemini.google.com/share/a408f2390abd
            values = values.repeat_interleave(group_size, -3) 
    else:
        assert num_groups == 1 or num_heads == num_groups # Might also be Multi-Query Attention
        # if num_groups == 1:
        #     keys = keys.expand(*leading_dims_k, num_heads, keys_len, keys_head_dim)
        #     values = values.expand(*leading_dims_k, num_heads, keys_len, keys_head_dim)

    factor = torch.rsqrt(query_head_dim) if scale is None else scale
    assert query_head_dim == keys_head_dim
    attn_scores = queries @ keys.transpose(-2, -1) * factor

    mask = causal_mask(query_len, keys_len, device=queries.device) if is_causal else attn_mask
    neg_inf = -torch.inf # torch.finfo(attn_scores.dtype).min # 
    if mask is not None:
        attn_scores.masked_fill_(mask, neg_inf)
    attn_weights = softmax(attn_scores, dim=-1)
    attn_weights = F.dropout(dropout_p) # Attn dropout
    assert keys_len == values.shape[-2]

    return attn_weights @ values

def scaled_dot_product_attention(**kawargs):
    F.scaled_dot_product_attention(**kawargs)

class GroupedQueryAttention(nn.Module):
    def __init__(self, *, cfg, d_in, d_out_qk, num_heads, num_kv_groups, d_out_v = None,
                 qk_norm=True, dtype=None, droput_p=0, qkv_bias=False):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        assert d_out_qk % num_heads == 0
        if d_out_v is None: d_out_v = d_out_qk
        else: assert d_out_v % num_heads == 0
        self.d_out_qk = d_out_qk
        self.d_out_v = d_out_v 

        self.head_dim_qk = d_out_qk // num_heads
        self.head_dim_v = d_out_v // num_heads

        self.W_query = nn.Linear(d_in, self.d_out_qk, bias=qkv_bias, dtype=dtype) # self.d_out_qk == group_size * num_kv_groups * self.head_dim_qk
        # self.W_kv = nn.Linear(d_in, 2 * num_kv_groups * out_head_dim, bias=qkv_bias, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim_qk, bias=qkv_bias, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim_v, bias=qkv_bias, dtype=dtype)

        self.out_proj = nn.Linear(self.d_out_v, d_in, bias=False, dtype=dtype)
        
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim_qk, eps=1e-6, dtype=dtype)
            self.k_norm = RMSNorm(self.head_dim_qk, eps=1e-6, dtype=dtype)
        else:
            self.q_norm = self.k_norm = None
        
        self.dropout_p = droput_p
        self.resid_dropout = nn.Dropout(droput_p)
        # cos, sin = compute_positional_params(
        #     head_dim=d_out_qk,
        #     theta_base=cfg["rope_base"],
        #     context_length=cfg["context_length"]
        # )
        # self.register_buffer("cos", cos, persistent=False)
        # self.register_buffer("sin", sin, persistent=False)

    def forward(self, x, cos, sin, *, offset=0, cache: KvCache = None, mask="causal", exact=False):
        *batch_dims, num_tokens, d_in = x.shape

        queries: torch.Tensor = self.W_query(x)  # (batch_size, num_tokens, num_heads * head_dim)
        keys: torch.Tensor = self.W_key(x)       # (batch_size, num_tokens, num_kv_groups * head_dim)
        values: torch.Tensor = self.W_value(x)   # Same as keys
        # keys_values = self.W_kv(x)

        queries = queries.view(*batch_dims, num_tokens, self.num_heads, self.head_dim_qk).transpose(-2, -3)
        keys = keys.view(*batch_dims, num_tokens, self.num_kv_groups, self.head_dim_qk).transpose(-2, -3)
        values = values.view(*batch_dims, num_tokens, self.num_kv_groups, self.head_dim_v).transpose(-2, -3)
        # keys, values = keys_values.view(*batch_dims, num_tokens, 2, self.num_kv_groups, self.out_head_dim) \
        #     .permute(2, 0, 3, 1, 4).unbind(0)

        if self.q_norm:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        # offset_slice = slice(int(offset), int(offset + num_tokens)) if isinstance(offset, int) \
        #     else [slice(int(i), int(i + num_tokens)) for i in offset]
        queries = apply_rotary_embedding(queries, cos, sin, offset=offset)
        keys = apply_rotary_embedding(keys, cos, sin, offset=offset)

        if self.group_size > 1:
            keys = keys.repeat_interleave(self.group_size, dim=-3)
            values = values.repeat_interleave(self.group_size, dim=-3)

            # batch_size, num_groups, num_tokens, out_head_dim = keys.shape
            # keys = keys = keys.unsqueeze(dim=2).expand(...)
            # keys = keys[:, :, None, :, :].expand(batch_size, num_groups, self.group_size, num_tokens, out_head_dim)
            # keys = keys.reshape(batch_size, self.num_heads, num_tokens, out_head_dim)
            # values = values[:, :, None, :, :].expand(batch_size, num_groups, self.group_size, num_tokens, out_head_dim)
            # values = values.reshape(batch_size, self.num_heads, num_tokens, out_head_dim)

        keys, values, _, mask = cache.update_and_fetch(
            keys, values, query_mask_len=num_tokens, mask=mask
        )

        backends = [SDPBackend.MATH] if exact else [b for b in SDPBackend if b.name not in {"ERROR", "OVERRIDEABLE"}]
        dropout_p = self.dropout_p if self.training else 0
        with sdpa_kernel(backends):
            context_vec = scaled_dot_product_attention(
                queries.contiguous(),
                keys.contiguous(),
                values.contiguous(),
                attn_mask=mask if mask != "causal" else None,
                is_causal=mask == "causal",
                dropout_p=dropout_p,
                enable_gqa=True
            )
        
        context = context.transpose(-2, -3).contiguous().view(*batch_dims, num_tokens, self.d_out_v)
        return self.resid_dropout(self.out_proj(context_vec)) # resid_dropout
