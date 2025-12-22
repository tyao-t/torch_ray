# "affine" quantize
import numpy as np

def quantize_affine_no_group(x, bits=4):
    x = np.asarray(x, dtype=np.float32)
    qmax = (1 << bits) - 1

    alpha = float(x.max())
    beta  = float(x.min())

    ## fall back case: all values are the same
    if alpha == beta:
        scale = 0.0
        bias = beta
        q = np.zeros_like(x, dtype=np.uint8)
        return q, scale, bias

    scale = (alpha - beta) / qmax
    bias = beta

    q = np.rint((x - bias) / scale)          # round
    q = np.clip(q, 0, qmax).astype(np.uint8) # clamp to [0, qmax]
    return q, scale, bias

def dequantize_affine_no_group(q, scale, bias):
    q = np.asarray(q, dtype=np.float32)
    return q * float(scale) + float(bias)

# MLX 的 q（也就是 w_q）不是逐元素数组，
# 而是 packed 的 bitstream（通常是 uint32），dequantize 必须知道每个元素占多少 bit 才能把它解出来。

# q会被打包进uint32 N/8 ceil[N * bits/32]个uint32
# biases和scales可能是stream所以还是要知道group_size

import numpy as np

def quantize_affine_2d_grouped(w, group_size=64, bits=4):
    w = np.asarray(w, dtype=np.float32)
    K, N = w.shape
    if N % group_size != 0:
        raise ValueError(f"N={N} must be divisible by group_size={group_size}")

    qmax = (1 << bits) - 1
    G = N // group_size

    q = np.empty((K, N), dtype=np.uint8)
    scales = np.empty((K, G), dtype=np.float32)
    biases = np.empty((K, G), dtype=np.float32)

    for k in range(K):
        row = w[k]
        for g in range(G):
            start = g * group_size
            end = start + group_size
            block = row[start:end]

            alpha = float(block.max())
            beta  = float(block.min())   # bias = min

            if alpha == beta:
                scale = 0.0
                qq = np.zeros((group_size,), dtype=np.uint8)
            else:
                scale = (alpha - beta) / qmax
                qq = np.rint((block - beta) / scale)
                qq = np.clip(qq, 0, qmax).astype(np.uint8)

            scales[k, g] = scale
            biases[k, g] = beta
            q[k, start:end] = qq

    return q, scales, biases


def dequantize_affine_2d_grouped(q, scales, biases, group_size=64):
    q = np.asarray(q, dtype=np.float32)
    scales = np.asarray(scales, dtype=np.float32)
    biases = np.asarray(biases, dtype=np.float32)

    K, N = q.shape
    if N % group_size != 0:
        raise ValueError(f"N={N} must be divisible by group_size={group_size}")

    num_groups = N // group_size
    if scales.shape != (K, num_groups) or biases.shape != (K, num_groups):
        raise ValueError(f"scales/biases must have shape {(K, num_groups)}, got {scales.shape} / {biases.shape}")

    w_hat = np.empty((K, N), dtype=np.float32)

    for k in range(K):
        for g in range(num_groups):
            start = g * group_size
            end = start + group_size
            w_hat[k, start:end] = q[k, start:end] * scales[k, g] + biases[k, g]

    return w_hat

from numpy.typing import NDArray
def quantized_matmul_unpacked(
    scales: NDArray[np.floating],
    biases: NDArray[np.floating],
    group_size: int,
    bits: int,
    a: NDArray[np.floating],
    q: NDArray[np.integer],
) -> NDArray[np.floating]:
    """
      a:      (M, N) float32
      q:      (K, N) int4 or int8 codes unpacked, each in [0, 2^bits - 1]
      scales: (K, G) float32, where G = N / group_size
      biases: (K, G) float32, where G = N / group_size
      out:    (M, K) float32 (same dtype as a)
    """
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

# static inline void check(bool cond, const std::string& msg) {
#     if (!cond) throw std::runtime_error(msg);
# }

# /**
#  * Quantized matmul (unpacked codes).
#  *
#  * a:      (M, N) float
#  * q:      (K, N) uint8 (or int8/uint8 codes), each in [0, 2^bits - 1]
#  * scales: (K, G) float, where G = N / group_size
#  * biases: (K, G) float, where G = N / group_size
#  *
#  * Returns:
#  *   out: (M, K) float
#  *
#  * Memory layout: row-major contiguous.
#  * Indexing:
#  *   a[m*N + n]
#  *   q[k*N + n]
#  *   scales[k*G + g]
#  *   biases[k*G + g]
#  *   out[m*K + k]
#  */
# std::vector<float> quantized_matmul_unpacked_cpp(
#     const float* a,                 // size M*N
#     const std::uint8_t* q,          // size K*N
#     const float* scales,            // size K*G
#     const float* biases,            // size K*G
#     int M, int N, int K,
#     int group_size,
#     int bits
# ) {
#     check(M > 0 && N > 0 && K > 0, "M,N,K must be positive");
#     check(group_size > 0, "group_size must be positive");
#     check(bits >= 1 && bits <= 16, "bits must be in [1,16]");
#     check(N % group_size == 0, "N must be divisible by group_size");

#     const int G = N / group_size;
#     const int qmax = (1 << bits) - 1;

#     // Optional: validate q range (O(K*N), can be removed for speed)
#     {
#         std::uint8_t qmin = 255, qmax_seen = 0;
#         for (int i = 0; i < K * N; ++i) {
#             qmin = std::min(qmin, q[i]);
#             qmax_seen = std::max(qmax_seen, q[i]);
#         }
#         check(qmin >= 0, "q contains negative? (uint8 shouldn't)");
#         check(qmax_seen <= qmax, "q contains values > (2^bits - 1)");
#     }

#     // out: (M, K)
#     std::vector<float> out((size_t)M * K, 0.0f);

#     // temp buffer for one group's dequantized weights: length group_size
#     std::vector<float> w_chunk((size_t)group_size);

#     // Loop order matches your Python structure:
#     // for i in range(K):
#     //   for j in range(0, N, group_size):
#     //     g = j/group_size
#     //     w_chunk[t] = q[i, j+t]*scale + bias
#     //     out[:, i] += A_block @ w_chunk
#     for (int i = 0; i < K; ++i) {
#         for (int g = 0; g < G; ++g) {
#             const int j0 = g * group_size;
#             const float scale = scales[i * G + g];
#             const float bias  = biases[i * G + g];

#             // dequantize this group's weights for row i
#             const std::uint8_t* q_row = q + (size_t)i * N + j0;
#             for (int t = 0; t < group_size; ++t) {
#                 w_chunk[t] = (float)q_row[t] * scale + bias;
#             }

#             // out[m, i] += dot(a[m, j0:j0+gs], w_chunk)
#             for (int m = 0; m < M; ++m) {
#                 const float* a_row = a + (size_t)m * N + j0;
#                 float sum = 0.0f;
#                 // dot product of length group_size
#                 for (int t = 0; t < group_size; ++t) {
#                     sum += a_row[t] * w_chunk[t];
#                 }
#                 out[(size_t)m * K + i] += sum;
#             }
#         }
#     }

#     return out;
# }