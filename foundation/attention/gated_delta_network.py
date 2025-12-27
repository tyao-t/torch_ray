import torch
from torch import nn
import torch.nn.functional as F

class GatedMultiHeadAttention(nn.Module):
    def __init__(
        self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False
    ):
        super().__init__()
        assert d_out % num_heads == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_gate = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
            persistent=False,
        )

    def forward(self, x):
        B, L, _ = x.shape
        queries = self.W_query(x)
        gate = self.W_gate(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        keys = keys.view(B, L, self.num_heads, self.head_dim)
        values = values.view(B, L, self.num_heads, self.head_dim)
        queries = queries.view(B, L, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:L, :L]
        attn_scores.masked_fill_(
            mask_bool, torch.finfo(attn_scores.dtype).min
        )

        attn_weights = torch.softmax(
            attn_scores / (self.head_dim ** 0.5), dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context = (attn_weights @ values).transpose(1, 2)
        context = context.reshape(B, L, self.d_out)

        context = context * torch.sigmoid(gate)
        out = self.out_proj(context)
        return out

class GatedDeltaNet(nn.Module):
    def __init__(
        self, d_in, d_out, dropout, num_heads, qkv_bias=False,
        # conv_size=4
    ):
        super().__init__()
        assert d_out % num_heads == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # self.conv_q = nn.Conv1d(d_out, d_out, kernel_size=conv_size, 
        #                         groups=d_out, padding=conv_size - 1) Same for conv_k conv_v

        # Output Gate: 类似于 LSTM/GRU 的输出门，控制最终输出多少信息
        self.W_gate = nn.Linear(d_in, d_out, bias=False)

        # Beta (Update Gate): 决定 "写入强度", (每个特征维度都有自己的 beta)
        self.W_beta = nn.Linear(d_in, d_out, bias=False)

        # Alpha (Decay Gate): 决定 "遗忘速度", 这里输出维度是 num_heads, 同一个 Head 里的所有特征共享同一个 alpha
        self.W_alpha = nn.Linear(d_in, num_heads, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(num_heads))

        # A 是 "Multi-Scale Memory" (多尺度记忆) 的关键, 形状是 (num_heads,)
        # A 大的 Head：健忘，只看短期 (Short-term context)
        # A 小的 Head：记性好，看长期 (Long-term context)
        A_init = torch.empty(num_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A_init))

        self.norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, _ = x.shape

        queries = self.W_query(x) # (B, L, D_out)
        keys = self.W_key(x)      # (B, L, D_out)
        values = self.W_value(x)  # (B, L, D_out)
        # queries = F.silu(self.conv_q(queries.transpose(-2, -3))[..., :L]).transpose(-2, -3) Same for K, V
        # Conv是对L这个维度进行 Conv
        gate = self.W_gate(x) # (B, L, D_out)
 

        beta = torch.sigmoid(self.W_beta(x)) # (B, L, D_out)

        # 论文公式 alpha = exp(A_log + W_alpha(x) + bias)
        alpha = (-self.A_log.exp().view(1, 1, -1) * F.softplus(
            self.W_alpha(x) + self.dt_bias
        )).exp() # (B, L, heads)

        keys = keys.view(B, L, self.num_heads, self.head_dim).transpose(-2, -3) # (B, num_heads, L, head_dim)
        values = values.view(B, L, self.num_heads, self.head_dim).transpose(-2, -3)
        queries = queries.view(B, L, self.num_heads, self.head_dim).transpose(-2, -3)
        beta = beta.view(B, L, self.num_heads, self.head_dim).transpose(-2, -3)
        gate = gate.view(B, L, self.num_heads, self.head_dim).transpose(-2, -3)

        queries = F.normalize(queries, dim=-1, p=2) / (self.head_dim ** 0.5)
        keys = F.normalize(keys, dim=-1, p=2)

        S = torch.zeros((B, self.num_heads, self.head_dim, self.head_dim), dtype=x.dtype, device=x.device)

        # Q @ K^T @ V -> Q @ (K^T @ V) 
        # 模拟后者的 (head_dim, L) @ (L, head_dim)
        # 计算 K^T @ V 时， 在计算 Key 的第i个特征和Value的第j个特征的相关性
        # 例如K的特征i很强，同时V的特征j也很强，那么S_{ij}就会很大。
        outs = []
        for t in range(L):
            k_t = keys[:, :, t] # (B, heads, head_dim)
            q_t = queries[:, :, t]  # (B, heads, head_dim)
            v_t = values[:, :, t] # (B, heads, head_dim)
            beta_t = beta[:, :, t] # (B, heads, head_dim)

            a_t = alpha[:, t].unsqueeze(-1).unsqueeze(-1) # (B, heads, 1, 1)

            S = S * a_t # (B, heads, head_dim, head_dim)

            # Retrieve (回忆), 当前的 key 去问 S 觉得现在的 value 应该是多少
            kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2) # (B, heads, head_dim)

            # Compute delta, 再乘 beta_t
            delta = (v_t - kv_mem) * beta_t # (B, heads, head_dim)

            # 外积更新,修正S矩阵，k_t ⊗ delta
            S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)

            # Query 最新的记忆
            y_t = (S * q_t.unsqueeze(-1)).sum(dim=-2) # (B, heads, head_dim)

            outs.append(y_t)

        context_vec = torch.stack(outs, dim=-1).transpose(-2, -3).contiguous() # (B, L, heads, head_dim)
        context_vec = self.norm(context_vec) * F.silu(gate)
        context = context.view(B, L, self.d_out)
        context = self.dropout(context)
        out = self.out_proj(context)
        return out
