import torch
import torch.nn as nn
from foundation.attention.multi_head_attn import causal_mask
from foundation.operators.rotary_pos_embeddings import compute_positional_params, apply_rotary_embedding
from foundation.operators.normalizations import RMSNorm
from inference.kv_cache import MLAKVCache

class MultiHeadLatentAttentionNaive(nn.Module):
    def __init__(self, d_in, d_out, dropout, num_heads,
                 qkv_bias=False, latent_dim=None):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.latent_dim = latent_dim if latent_dim is not None else max(16, d_out // 8)

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias) 
        self.W_DKV = nn.Linear(d_in, self.latent_dim, bias=qkv_bias)
        self.W_UKV = nn.Linear(self.latent_dim, 2*d_out, bias=qkv_bias)
        # self.W_UK = nn.Linear(self.latent_dim, d_out, bias=qkv_bias)
        # self.W_UV = nn.Linear(self.latent_dim, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, offset, kv_cache: MLAKVCache = None, mask="causal"):
        B, num_new_tokens, _ = x.shape
        num_heads = self.num_heads
        head_dim = self.head_dim

        queries_all = self.W_query(x)
        latenL_new = self.W_DKV(x)

        if kv_cache is None:
            latent_total = latenL_new
        else:
            latent_total, _, _, _ = kv_cache.update_and_fetch_all(latent_vector=latenL_new, rope_key=None, mask=mask)

        # keys_all = self.W_UK(latent_total)   # (batch, L_k_total, d_out)
        # values_all = self.W_UV(latent_total)   # (batch, L_k_total, d_out)
        keys_all, values_all = torch.chunk(self.W_UKV(latent_total), chunks=2, dim=-1)

        queries = queries_all.view(B, num_new_tokens, num_heads, head_dim).transpose(-2, -3)
        total_kv_tokens = keys_all.shape[-3]
        keys = keys_all.view(B, total_kv_tokens, num_heads, head_dim).transpose(-2, -3)
        values = values_all.view(B, total_kv_tokens, num_heads, head_dim).transpose(-2, -3)

        attn_scores = queries @ keys.transpose(-1, -2)

        mask_bool = causal_mask(num_new_tokens, total_kv_tokens, device=queries.device)
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores * torch.rsqrt(keys.shape[-1]), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(-2, -3)

        context_vec = context_vec.contiguous().view(B, num_new_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec

class DeepSeekV3LatentAttention(nn.Module):
    def __init__(self, cfg, d_in, d_out, num_heads, 
                 rope_dim=64, latent_dim=512, dropout=0.0, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.d_out = d_out
        self.latent_dim = latent_dim
        self.rope_dim = rope_dim
        self.q_content_dim = d_out
        
        assert d_out % num_heads == 0
        self.head_dim = d_out // num_heads 

        self.W_DKV = nn.Linear(d_in, latent_dim, bias=qkv_bias)
        self.kv_norm = RMSNorm(latent_dim)
        self.W_KRope = nn.Linear(d_in, num_heads * rope_dim, bias=qkv_bias)

        # Also note: DeepSeek V3 normally uses a larger latent dim for Q (e.g., 1536) than KV (512)
        # Training Path, where we will also compress Q
        self.W_DQ = nn.Linear(d_in, latent_dim, bias=qkv_bias)
        self.q_norm = RMSNorm(latent_dim)
        self.W_UQ = nn.Linear(latent_dim, d_out, bias=False)
        self.W_QRope = nn.Linear(d_in, num_heads * rope_dim, bias=qkv_bias)

        # Inference path, absorbed weight: W_DQ * W_UQ merged, W_QR concatenated
        self.W_Q = nn.Linear(d_in, d_out + num_heads * rope_dim, bias=qkv_bias)

        self.W_UK = nn.Linear(latent_dim, d_out, bias=False)
        self.W_UV = nn.Linear(latent_dim, d_out, bias=False)

        self.out_proj = nn.Linear(d_out, d_in)

        cos, sin = compute_positional_params(
            head_dim=rope_dim, 
            theta_base=cfg.get("rope_base", 10000), 
            context_length=cfg.get("context_length", 4096)
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x, offset, kv_cache: MLAKVCache = None, mask=None):
        B, num_new_tokens, _ = x.shape
        
        c_kv = self.kv_norm(self.W_DKV(x)) # (B, L_new, latent_dim)
        
        k_rope = self.W_KRope(x).view(B, num_new_tokens, self.num_heads, self.rope_dim).transpose(-2, -3)
        
        if self.training:
            q_latent = self.W_DQ(x) 
            q_latent = self.q_norm(q_latent) 
            q_content = self.W_UQ(q_latent).view(B, num_new_tokens, self.num_heads, self.head_dim).transpose(-2, -3)
            q_rope = self.W_QRope(x).view(B, num_new_tokens, self.num_heads, self.rope_dim).transpose(-2, -3)
        else:
            # Inference: No latent/Norm step
            q_total = self.W_Q(x)
            q_content = q_total[:, :, :self.q_content_dim].contiguous().view(B, num_new_tokens, self.num_heads, self.head_dim).transpose(-2, -3)
            q_rope = q_total[:, :, self.q_content_dim:].contiguous().view(B, num_new_tokens, self.num_heads, self.rope_dim).transpose(-2, -3)

        # Apply RoPE (Assuming apply_rotary_embedding is imported)
        q_rope = apply_rotary_embedding(q_rope, self.cos, self.sin, offset=offset)
        k_rope = apply_rotary_embedding(k_rope, self.cos, self.sin, offset=offset)

        if kv_cache is not None:
            # (B, total_kv_tokens, latent_dim), (B, num_heads, total_kv_tokens, rope_dim)
            c_kv_history, k_rope_history, _, _ = kv_cache.update_and_fetch_all(
                latent_vector=c_kv, 
                rope_key=k_rope,
                mask=mask
            )
        else:
            c_kv_history = c_kv
            k_rope_history = k_rope

        # C @ (A @ B) ^ T = C @ B^T @ A^T
        w_uk = self.W_UK.weight.view(self.num_heads, self.head_dim, self.latent_dim)

        # (B, H, N_q, D_h) @ (B, H, N_kv, D_h)^T
        # O(N_q * D_h * N_kv) + O(N_kv * D_h * D_l) vs O(N_q * D_h * D_l) + O(N_q * D_l * N_kv)
        q_content_absorbed = torch.matmul(q_content, w_uk) # (B, num_heads, L_q, latent_dim)

        score_content = q_content_absorbed @ c_kv_history.unsqueeze(1).transpose(-1, -2) # (B, num_heads, L_q, total_kv_tokens)

        score_rope = q_rope @ k_rope_history.transpose(-1, -2) # Same shape as score_content
        
        scores = (score_content + score_rope) * torch.rsqrt(self.head_dim + self.rope_dim)
        
        scores.masked_fill_(causal_mask(num_new_tokens, c_kv_history.shape[-2], device=x.device), -torch.inf)
        attn_weights = torch.softmax(scores, dim=-1) # self.dropout(torch.softmax(scores, dim=-1))
        
        context_latent = torch.matmul(attn_weights, c_kv_history.unsqueeze(1)) # (B, num_heads, L_new, latent_dim)

        # O(L_new * L_total * Full_Dim + L_total * latent_dim * Full_dim) vs 
        # O(L_new * L_total * latent_dim + L_new * latent_dim * Full_dim)  

        w_uv_transposed = self.W_UV.weight.view(self.num_heads, self.head_dim, self.latent_dim).transpose(-1, -2)
        context_vec = torch.matmul(context_latent, w_uv_transposed)
        # Result: (B, num_heads, L_new, head_dim)
        
        context_vec = context_vec.transpose(-2, -3).contiguous().view(B, num_new_tokens, self.d_out)
        return self.out_proj(context_vec)
