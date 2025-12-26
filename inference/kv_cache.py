import torch
from typing import Optional, Tuple, Union
from foundation.attention.multi_head_attn import causal_mask
from abc import ABC, abstractmethod

class KvCache(ABC):
    @abstractmethod
    def update_and_fetch(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        query_mask_len: int | None = None,
        mask: torch.Tensor | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, Optional[torch.Tensor]]:
        # returns keys, values, offset, mask
        pass

class FullKvCache(KvCache): 
    def __init__(self):
        self.keys = None
        self.values = None
        self.offset: int = 0

    def update_and_fetch(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        query_mask_len: int | None = None,
        mask: torch.Tensor | str | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, Optional[torch.Tensor]]:
        
        if self.key_values is None:
            assert self.offset == 0
            self.keys, self.values = key, value
            batch_size, num_heads, num_tokens, head_dim = key.shape
            self.offset = num_tokens
            return key, value, self.offset, mask
        else:
            batch_size, num_heads, num_tokens, head_dim = key.shape
            assert key.shape == value.shape
            
            assert self.keys.shape == (batch_size, num_heads, self.offset, head_dim)
            assert self.values.shape  == (batch_size, num_heads, self.offset, head_dim)

            self.keys = torch.cat([self.keys, key], dim=-2)
            self.values = torch.cat([self.values, value], dim=-2)
            
            self.offset += num_tokens
            
            return self.keys, self.values, self.offset, mask

    def rewind(self, n: int):
        if n == 0 or self.keys is None:
            return

        if n >= self.offset:
            self.reset()
            return
        
        self.offset -= n
        if self.key_values is not None:
            self.keys = self.keys[:, :, :self.offset, :]
            self.values = self.values[:, :, :self.offset, :]

    def reset(self):
        self.keys = self.values = None
        self.offset = 0

class BatchingKvCache(KvCache):
    def __init__(self, max_active_requests: int):
        self.max_active_requests = max_active_requests
        self.kv_caches = [None] * max_active_requests
        # Optimize Batching KV Cache: Paged Attention: https://chatgpt.com/share/693ef34f-adec-8003-87f4-c5e5d5a2c591

    def update_and_fetch(
        self,
        keys,
        values,
        query_mask_len=None, # int | None 
        mask = None #torch.tensor | str | None
    ):
        assert keys.shape == values.shape
        batch_size, num_heads, N, head_dim = keys.shape
        assert batch_size == self.max_active_requests

        data = []
        for b in range(batch_size):
            if self.kv_caches[b] is None:
                data.append(None)
                continue
            key, value = keys[b : b + 1], values[b : b + 1]
            new_key, new_value, seq_len, mask = self.kv_caches[b].update_and_fetch(
                key, value
            )
            data.append((new_key[0], new_value[0], seq_len, mask))

        max_seq_len = max((0 if d is None else d.shape[-2] for d in data)) # Optimize: Paged Attention (KV Cache Paging in GPU HBM)
        keys = torch.zeros((self.max_active_requests, num_heads, max_seq_len, head_dim), dtype=key.dtype, device=key.device)
        values = torch.zeros((self.max_active_requests, num_heads, max_seq_len, head_dim), dtype=value.dtype, device=value.device)
        masks = torch.full(
            (self.max_active_requests, query_mask_len, max_seq_len), 
            -torch.inf,
            dtype=key.dtype,
            device=key.device
        )
        for b in range(batch_size):
            # if data[b] is None:
            #     # for some reasons we need to do this, otherwise it will cause wrong output?
            #     # maybe precision issues?
            #     masks[b, :, :] = causal_mask(query_mask_len, max_seq_len, dtype=key.dtype)
            #     continue
            
            key, value, seq_len, mask = data[b]
            keys[b, :, max_seq_len - seq_len : max_seq_len, :] = key
            values[b, :, max_seq_len - seq_len : max_seq_len, :] = value
            
            if mask is None:
                mask[b:, :, max_seq_len - seq_len : max_seq_len] = 0
            elif mask == "causal":
                masks[b, :, max_seq_len - seq_len : max_seq_len] = causal_mask(
                    query_mask_len, seq_len, device=keys
                )
            elif isinstance(mask, torch.Tensor):
                masks[b, :, max_seq_len - seq_len : max_seq_len] = mask

        return keys, values, None, masks.reshape(batch_size, 1, query_mask_len, max_seq_len)

    def add_request(self, prefilled, id: int):
        assert id < self.max_active_requests
        self.kv_caches[id] = prefilled

    def remove_request(self, id: int):
        if self.kv_caches is None:
            raise ValueError(f"Request id {id} is not in the cache")
        self.kv_caches[id] = None

class RotatingKvCache(KvCache):
    def __init__(self, capacity):
        self.offset = 0
        self.keys = None
        self.values = None
        self.capacity = capacity

    def update_and_fetch(self, key, value, query_mask_len=None, mask=None):
        batch_size, num_heads, num_tokens, head_dim = key.shape
        assert num_tokens <= self.capacity 

        if self.keys is None:
            assert self.offset == 0
            self.keys = torch.zeros((batch_size, num_heads, self.capacity, head_dim))
            self.values = torch.zeros((batch_size, num_heads, self.capacity, head_dim))
            self.keys[:, :, :num_tokens, :] = key
            self.values[:, :, :num_tokens, :] = value
            self.offset = num_tokens
            return self.keys[:self.offset], self.values[:self.offset], self.offset, mask

        if self.offset + num_tokens <= self.capacity:
            self.keys[:, :, self.offset:self.offset+num_tokens, :] = key
            self.values[:, :, self.offset:self.offset+num_tokens, :] = value
            self.offset += num_tokens
            return self.keys[:self.offset], self.values[:self.offset], self.offset, mask
        
        overflow = self.offset + num_tokens - self.capacity
        self.keys[:, :, :-overflow, :] = self.keys.copy()[:, :, overflow:, :]
        self.values[:, :, :-overflow, :] = self.values.copy()[:, :, overflow:, :]
        self.keys[:, :, -num_tokens:, :] = key
        self.values[:, :, -num_tokens:, :] = value
        self.offset = self.offset + num_tokens - self.capacity
        return self.keys[:self.offset], self.values[:self.offset], self.offset, mask

class MLAKVCache:
    def __init__(self):
        # Shape: [Batch, Tokens, Latent_Dim]
        self.latent_cache: Optional[torch.Tensor] = None 
        
        # Shape: [Batch, Heads, Tokens, Rope_Dim]
        self.rope_cache: Optional[torch.Tensor] = None   
        
        self.offset = 0
        
    def update_and_fetch(
        self,
        latent_vector: torch.Tensor,             # [Batch, New_Tokens, Latent_Dim]
        rope_key: Optional[torch.Tensor] = None, # [Batch, Heads, New_Tokens, Rope_Dim]
        mask: torch.Tensor | str | None = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, Optional[torch.Tensor]]:
        
        # Initialize on first pass
        if self.latent_cache is None:
            self.latent_cache = latent_vector
            self.rope_cache = rope_key
            self.offset = latent_vector.shape[1] # num_new_tokens
        else:
            self.latent_cache = torch.cat([self.latent_cache, latent_vector], dim=1)
            
            if rope_key is not None:
                self.rope_cache = torch.cat([self.rope_cache, rope_key], dim=2)
            
            self.offset += latent_vector.shape[1]
            
        return self.latent_cache, self.rope_cache, self.offset, mask

    def rewind(self, n: int):
        if n == 0 or self.latent_cache is None:
            return

        if n >= self.offset:
            self.reset()
            return
        
        self.offset -= n
        
        self.latent_cache = self.latent_cache[:, :self.offset, :]
        
        if self.rope_cache is not None:
            self.rope_cache = self.rope_cache[:, :, :self.offset, :]

    def reset(self):
        self.latent_cache = None
        self.rope_cache = None
        self.offset = 0

# class KVCache:
#     def __init__(self, n_layers, max_len, num_kv_groups, head_dim, device, dtype):
#         self.k = [None] * n_layers
#         self.v = [None] * n_layers
#         self.len = [0] * n_layers
#         self.max_len = max_len
#         self.num_kv_groups = num_kv_groups
#         self.head_dim = head_dim
#         self.device = device
#         self.dtype = dtype

#     def allocate(self, layer_idx, b):
#         if self.k[layer_idx] is None:
#             self.k[layer_idx] = torch.empty(b, self.num_kv_groups, self.max_len, self.head_dim,
#                                             device=self.device, dtype=self.dtype)
#             self.v[layer_idx] = torch.empty(b, self.num_kv_groups, self.max_len, self.head_dim,
#                                             device=self.device, dtype=self.dtype)
#             self.len[layer_idx] = 0

#     def append(self, layer_idx, k_new, v_new):
#         L = self.len[layer_idx]
#         T = k_new.shape[2]
#         self.k[layer_idx][:, :, L:L+T, :].copy_(k_new)
#         self.v[layer_idx][:, :, L:L+T, :].copy_(v_new)
#         self.len[layer_idx] = L + T

#     def view(self, layer_idx):
#         L = self.len[layer_idx]
#         return self.k[layer_idx][:, :, :L, :], self.v[layer_idx][:, :, :L, :]

#     def reset(self):
#         for i in range(len(self.k)):
#             self.k[i] = self.v[i] = None
#             self.len[i] = 0
