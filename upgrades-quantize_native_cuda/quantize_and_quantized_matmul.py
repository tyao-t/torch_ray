# "affine" quantize
import torch
import numpy as np
from numpy.typing import NDArray

def quantize_affine_no_group(x, bits=4):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    else:
        x = x.float()

    qmax = (1 << bits) - 1

    alpha = x.max()
    beta  = x.min()

    if alpha == beta:
        scale = 0.0
        bias = beta.item()
        q = torch.zeros_like(x, dtype=torch.uint8)
        return q, scale, bias

    scale = (alpha - beta) / qmax
    bias = beta

    q = torch.round((x - bias) / scale) # round 
    q = torch.clamp(q, 0, qmax).to(torch.uint8) # clamp to [0, qmax]
    
    return q, scale.item(), bias.item()

def dequantize_affine_no_group(q, scale, bias):
    if not isinstance(q, torch.Tensor):
        q = torch.tensor(q, dtype=torch.float32)
    else:
        q = q.float()

    return q * scale + bias

# MLX 的 q（也就是 w_q）不是逐元素数组，
# 而是 packed 的 bitstream（通常是 uint32），dequantize 必须知道每个元素占多少 bit 才能把它解出来。

# q会被打包进uint32 N/8 ceil[N * bits/32]个uint32
# biases和scales可能是stream所以还是要知道group_size

# def quantize_affine_2d_grouped(w, group_size=64, bits=4):
#     w = np.asarray(w, dtype=np.float32)
#     K, N = w.shape
#     if N % group_size != 0:
#         raise ValueError(f"N={N} must be divisible by group_size={group_size}")

#     qmax = (1 << bits) - 1
#     G = N // group_size

#     q = np.empty((K, N), dtype=np.uint8)
#     scales = np.empty((K, G), dtype=np.float32)
#     biases = np.empty((K, G), dtype=np.float32)

#     for k in range(K):
#         row = w[k]
#         for g in range(G):
#             start = g * group_size
#             end = start + group_size
#             block = row[start:end]

#             alpha = float(block.max())
#             beta  = float(block.min())   # bias = min

#             if alpha == beta:
#                 scale = 0.0
#                 qq = np.zeros((group_size,), dtype=np.uint8)
#             else:
#                 scale = (alpha - beta) / qmax
#                 qq = np.rint((block - beta) / scale)
#                 qq = np.clip(qq, 0, qmax).astype(np.uint8)

#             scales[k, g] = scale
#             biases[k, g] = beta
#             q[k, start:end] = qq

#     return q, scales, biases

# def dequantize_affine_2d_grouped(q, scales, biases, group_size=64):
#     q = np.asarray(q, dtype=np.float32)
#     scales = np.asarray(scales, dtype=np.float32)
#     biases = np.asarray(biases, dtype=np.float32)

#     K, N = q.shape
#     if N % group_size != 0:
#         raise ValueError(f"N={N} must be divisible by group_size={group_size}")

#     num_groups = N // group_size
#     if scales.shape != (K, num_groups) or biases.shape != (K, num_groups):
#         raise ValueError(f"scales/biases must have shape {(K, num_groups)}, got {scales.shape} / {biases.shape}")

#     w_hat = np.empty((K, N), dtype=np.float32)

#     for k in range(K):
#         for g in range(num_groups):
#             start = g * group_size
#             end = start + group_size
#             w_hat[k, start:end] = q[k, start:end] * scales[k, g] + biases[k, g]

#     return w_hat

def quantize_affine_2d_grouped(w, group_size=64, bits=4):
    if not isinstance(w, torch.Tensor):
        w = torch.tensor(w, dtype=torch.float32)
    else:
        w = w.float()
        
    K, N = w.shape
    if N % group_size != 0:
        raise ValueError(f"N={N} must be divisible by group_size={group_size}")

    qmax = (1 << bits) - 1
    G = N // group_size

    w_view = w.view(K, G, group_size)

    alpha = w_view.amax(dim=-1, keepdim=True) # (K, G, 1)
    beta  = w_view.amin(dim=-1, keepdim=True) # (K, G, 1)
    
    raw_scale = (alpha - beta) / qmax # (K, G, 1)
    
    scale_safe = torch.where(raw_scale == 0, torch.ones_like(raw_scale), raw_scale)

    q = torch.round((w_view - beta) / scale_safe)
    q = torch.clamp(q, 0, qmax).to(torch.uint8)

    q = q.view(K, N)
    scales = raw_scale.squeeze(-1)
    biases = beta.squeeze(-1)

    return q, scales, biases

def dequantize_affine_2d_grouped(q, scales, biases, group_size=64):
    if not isinstance(q, torch.Tensor):
        q = torch.tensor(q, dtype=torch.float32)
    else:
        q = q.float()

    if not isinstance(scales, torch.Tensor):
        scales = torch.tensor(scales, dtype=torch.float32, device=q.device)
    
    if not isinstance(biases, torch.Tensor):
        biases = torch.tensor(biases, dtype=torch.float32, device=q.device)

    K, N = q.shape
    G = N // group_size

    q_view = q.view(K, G, group_size)
    scales_view = scales.view(K, G, 1)
    biases_view = biases.view(K, G, 1)

    w = q_view * scales_view + biases_view

    return w.view(K, N)

def quantized_matmul_unpacked_torch(
    scales: torch.Tensor,
    biases: torch.Tensor,
    group_size: int,
    bits: int,
    a: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    assert a.dim() == 2 and q.dim() == 2
    M, N = a.shape
    K, Nq = q.shape
    assert Nq == N
    assert N % group_size == 0
    G = N // group_size
    assert scales.shape == (K, G) and biases.shape == (K, G)

    a = a.float()
    q = q.to(dtype=torch.int32)
    scales = scales.float()
    biases = biases.float()

    out = torch.zeros((M, K), dtype=torch.float32, device=a.device)

    for i in range(K):
        for j in range(0, N, group_size):
            g = j // group_size
            
            q_chunk = q[i, j:j+group_size]

            scale = scales[i, g]
            bias = biases[i, g]
            w_chunk = q_chunk.float() * scale + bias
            
            a_chunk = a[:, j:j+group_size]

            out[:, i] += a_chunk @ w_chunk

    return out

def quantized_matmul_elementwise_torch(
    scales: torch.Tensor,
    biases: torch.Tensor,
    group_size: int,
    bits: int,
    a: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    M, N = a.shape
    K, _ = q.shape
    
    a = a.float()
    q = q.float() 
    scales = scales.float()
    biases = biases.float()
    
    out = torch.zeros((M, K), dtype=torch.float32, device=a.device)
    
    for i in range(K):
        for j in range(N):
            g = j // group_size
            
            scale = scales[i, g]
            bias = biases[i, g]
            
            w_val = q[i, j] * scale + bias
            
            out[:, i] += a[:, j] * w_val
            
    return out

def quantized_matmul_elementwise_torch(
    scales: torch.Tensor,
    biases: torch.Tensor,
    group_size: int,
    bits: int,
    a: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    M, N = a.shape
    K, _ = q.shape
    
    G = scales.shape[1]

    a = a.float()
    q = q.float() 
    scales = scales.float()
    biases = biases.float()
    
    out = torch.zeros((M, K), dtype=torch.float32, device=a.device)
    
    for i in range(K):
        for g in range(G):
            scale = scales[i, g]
            bias = biases[i, g]
            g_index = g*group_size
            for j in range(group_size):
                col_idx = g_index+j         
                w_val = q[i, col_idx] * scale + bias
                out[:, i] += a[:, col_idx] * w_val
            
    return out

def quantized_matmul_elementwise_numpy(
    scales: NDArray[np.floating],
    biases: NDArray[np.floating],
    group_size: int,
    bits: int,
    a: NDArray[np.floating],
    q: NDArray[np.integer],
) -> NDArray[np.floating]:
    
    M, N = a.shape
    K, _ = q.shape
    
    a_f32 = a.astype(np.float32, copy=False)
    scales_f32 = scales.astype(np.float32, copy=False)
    biases_f32 = biases.astype(np.float32, copy=False)
    
    out = np.zeros((M, K), dtype=np.float32)
    
    for i in range(K):
        for j in range(N):
            g = j // group_size
            
            scale = scales_f32[i, g]
            bias = biases_f32[i, g]
            
            w_val = float(q[i, j]) * scale + bias
            
            out[:, i] += a_f32[:, j] * w_val

    return out

def quantized_matmul_unpacked(
    scales: NDArray[np.floating],
    biases: NDArray[np.floating],
    group_size: int,
    bits: int,
    a: NDArray[np.floating],
    q: NDArray[np.integer],
) -> NDArray[np.floating]:
    # a:      (M, N) float32
    # q:      (K, N) int4 or int8 codes unpacked, each in [0, 2^bits - 1]
    # scales: (K, G) float32, where G = N / group_size
    # biases: (K, G) float32, where G = N / group_size
    # out:    (M, K) float32 (same dtype as a)
    assert a.ndim == 2 and q.ndim == 2
    assert scales.ndim == 2 and biases.ndim == 2

    M, N = a.shape
    K, Nq = q.shape
    assert Nq == N

    assert isinstance(group_size, int) and group_size > 0
    assert isinstance(bits, int) and 1 <= bits <= 16

    assert N % group_size == 0
    G = N // group_size

    assert scales.shape == (K, G) and biases.shape == (K, G)

    qmax = (1 << bits) - 1
    q_min = int(np.min(q))
    q_max = int(np.max(q))
    assert 0 <= q_min and q_max <= qmax

    a_f32 = a.astype(np.float32, copy=False)
    scales_f32 = scales.astype(np.float32, copy=False)
    biases_f32 = biases.astype(np.float32, copy=False)

    out = np.zeros((M, K), dtype=np.float32)

    for i in range(K):
        for j in range(0, N, group_size):
            g = j // group_size
            scale = scales_f32[i, g]
            bias = biases_f32[i, g]
            w_chunk = q[i, j:j+group_size].astype(np.float32) * scale + bias # (group_size,)
            
            # for k in range(M):
            #     a_chunk = a_f32[k, j:j+group_size].astype(np.float32)
            #     out[k, i] += np.dot(a_chunk, w_chunk) 
            out[:, i] += a_f32[:, j:j+group_size] @ w_chunk # (M,)

    return out.astype(a.dtype, copy=False)
