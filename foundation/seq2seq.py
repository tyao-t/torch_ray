import torch.nn as nn
from foundation.feedforward.feedforward import FeedForward
from foundation.attention.multi_head_attn import MultiHeadAttention
import torch
class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(emb_dim)
        self.norm_2 = nn.LayerNorm(emb_dim)
        self.attn = MultiHeadAttention(d_in=emb_dim, d_out=emb_dim, \
            num_num_heads=num_heads, dropout=dropout)
        self.ff = FeedForward(emb_dim, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        attn_output = self.attn(x, x, x, mask=src_mask)
        attn_output = self.dropout_1(attn_output)
        x = x + attn_output
        x = self.norm_1(x)
        ff_output = self.ff(x)
        ff_output = self.dropout_2(ff_output)
        x = x + ff_output
        x = self.norm_2(x)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(emb_dim)
        self.norm_2 = nn.LayerNorm(emb_dim)
        self.norm_3 = nn.LayerNorm(emb_dim)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(num_heads, emb_dim, dropout=dropout)
        self.attn_2 = MultiHeadAttention(num_heads, emb_dim, dropout=dropout)
        self.ff = FeedForward(emb_dim, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, dst_mask):
        attn_output_1 = self.attn_1(x, x, x, mask=dst_mask)
        attn_output_1 = self.dropout_1(attn_output_1)
        x = x + attn_output_1
        x = self.norm_1(x)
        attn_output_2 = self.attn_2(x, e_outputs, e_outputs, mask=src_mask)
        attn_output_2 = self.dropout_2(attn_output_2)
        x = x + attn_output_2
        x = self.norm_2(x)

        ff_output = self.ff(x)
        ff_output = self.dropout_3(ff_output)
        x = x + ff_output
        x = self.norm_3(x)

        return x
class SinusoidalAbsolutePositionalEncoder(nn.Module):
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

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_layers, num_heads, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.pos_encoder = SinusoidalAbsolutePositionalEncoder(emb_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([EncoderLayer(emb_dim, num_heads, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, src, src_mask):
        x = self.embed(src)
        x = self.pos_encoder(x)
        for i in range(self.num_layers):
            x = self.layers[i](x, src_mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_layers, num_heads, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.pos_encoder = SinusoidalAbsolutePositionalEncoder(emb_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([DecoderLayer(emb_dim, num_heads, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, dst, e_outputs, src_mask, dst_mask):
        x_dst = self.embed(dst)
        x_dst = self.pos_encoder(x_dst)
        for i in range(self.num_layers):
            x_dst = self.layers[i](x_dst, e_outputs, src_mask, dst_mask)
        return self.norm(x_dst)
    
class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, dst_vocab, emb_dim, num_layers, num_heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, emb_dim, num_layers, num_heads, dropout)
        self.decoder = Decoder(dst_vocab, emb_dim, num_layers, num_heads, dropout)
        self.out = nn.Linear(emb_dim, dst_vocab)

    def forward(self, src, dst, src_mask=None, dst_mask="causal"):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(dst, e_outputs, src_mask, dst_mask)
        output = self.out(d_output)
        return output
