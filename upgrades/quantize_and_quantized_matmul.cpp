static inline void check(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

/**
 * Quantized matmul (unpacked codes).
 *
 * a:      (M, N) float
 * q:      (K, N) uint8 (or int8/uint8 codes), each in [0, 2^bits - 1]
 * scales: (K, G) float, where G = N / group_size
 * biases: (K, G) float, where G = N / group_size
 *
 * Returns:
 *   out: (M, K) float
 *
 * Memory layout: row-major contiguous.
 * Indexing:
 *   a[m*N + n]
 *   q[k*N + n]
 *   scales[k*G + g]
 *   biases[k*G + g]
 *   out[m*K + k]
 */
std::vector<float> quantized_matmul_unpacked_cpp(
    const float* a,                 // size M*N
    const std::uint8_t* q,          // size K*N
    const float* scales,            // size K*G
    const float* biases,            // size K*G
    int M, int N, int K,
    int group_size,
    int bits
) {
    check(M > 0 && N > 0 && K > 0, "M,N,K must be positive");
    check(group_size > 0, "group_size must be positive");
    check(bits >= 1 && bits <= 16, "bits must be in [1,16]");
    check(N % group_size == 0, "N must be divisible by group_size");

    const int G = N / group_size;
    const int qmax = (1 << bits) - 1;

    // Optional: validate q range (O(K*N), can be removed for speed)
    {
        std::uint8_t qmin = 255, qmax_seen = 0;
        for (int i = 0; i < K * N; ++i) {
            qmin = std::min(qmin, q[i]);
            qmax_seen = std::max(qmax_seen, q[i]);
        }
        check(qmin >= 0, "q contains negative? (uint8 shouldn't)");
        check(qmax_seen <= qmax, "q contains values > (2^bits - 1)");
    }

    // out: (M, K)
    std::vector<float> out((size_t)M * K, 0.0f);

    // temp buffer for one group's dequantized weights: length group_size
    std::vector<float> w_chunk((size_t)group_size);

    // Loop order matches your Python structure:
    // for i in range(K):
    //   for j in range(0, N, group_size):
    //     g = j/group_size
    //     w_chunk[t] = q[i, j+t]*scale + bias
    //     out[:, i] += A_block @ w_chunk
    for (int i = 0; i < K; ++i) {
        for (int g = 0; g < G; ++g) {
            const int j0 = g * group_size;
            const float scale = scales[i * G + g];
            const float bias  = biases[i * G + g];

            // dequantize this group's weights for row i
            const std::uint8_t* q_row = q + (size_t)i * N + j0;
            for (int t = 0; t < group_size; ++t) {
                w_chunk[t] = (float)q_row[t] * scale + bias;
            }

            // out[m, i] += dot(a[m, j0:j0+gs], w_chunk)
            for (int m = 0; m < M; ++m) {
                const float* a_row = a + (size_t)m * N + j0;
                float sum = 0.0f;
                // dot product of length group_size
                for (int t = 0; t < group_size; ++t) {
                    sum += a_row[t] * w_chunk[t];
                }
                out[(size_t)m * K + i] += sum;
            }
        }
    }

    return out;
}

#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>

static inline void check(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

// Extract bits starting at (word_idx, bit_off) from a packed uint32 stream.
// Returns the next code in [0, (1<<bits)-1] and advances (word_idx, bit_off).
static inline std::uint32_t unpack_next_code(
    const std::uint32_t* row_words,
    int words_per_row,
    int bits,
    std::uint32_t mask,
    int& word_idx,
    int& bit_off
) {
    // We need bits from the current word; may straddle into the next word.
    // row_words[word_idx] contains bits [0..31] of that word (LSB-first).
    check(word_idx >= 0 && word_idx < words_per_row, "packed q: word_idx OOB");

    std::uint32_t w0 = row_words[word_idx];
    std::uint32_t v = w0 >> bit_off;

    const int used = 32 - bit_off;
    if (used < bits) {
        // Need some bits from the next word
        check(word_idx + 1 < words_per_row, "packed q: needs next word but row ended");
        std::uint32_t w1 = row_words[word_idx + 1];
        v |= (w1 << used);
    }

    std::uint32_t code = v & mask;

    // advance bit cursor
    bit_off += bits;
    while (bit_off >= 32) {
        bit_off -= 32;
        word_idx += 1;
    }
    return code;
}

/**
 * Quantized matmul (packed bitstream codes).
 *
 * a:        (M, N) float
 * q_packed: (K, W) uint32, where W = ceil(N * bits / 32)
 * scales:   (K, G) float, where G = N / group_size
 * biases:   (K, G) float, where G = N / group_size
 *
 * Returns out: (M, K) float
 *
 * Layout: row-major contiguous.
 * Indexing:
 *   a[m*N + n]
 *   q_packed[k*W + w]
 *   scales[k*G + g]
 *   biases[k*G + g]
 *   out[m*K + k]
 */
std::vector<float> quantized_matmul_packed_cpp(
    const float* a,                   // size M*N
    const std::uint32_t* q_packed,     // size K*W
    const float* scales,              // size K*G
    const float* biases,              // size K*G
    int M, int N, int K,
    int group_size,
    int bits
) {
    check(M > 0 && N > 0 && K > 0, "M,N,K must be positive");
    check(group_size > 0, "group_size must be positive");
    check(bits >= 1 && bits <= 16, "bits must be in [1,16]");
    check(N % group_size == 0, "N must be divisible by group_size");

    const int G = N / group_size;
    const std::uint32_t mask = (bits == 32) ? 0xFFFFFFFFu : ((1u << bits) - 1u);

    // Number of 32-bit words per packed row (K rows, each row packs N codes)
    const int W = (N * bits + 31) / 32;

    // out: (M, K)
    std::vector<float> out((size_t)M * K, 0.0f);

    // temp buffer for one group's dequantized weights
    std::vector<float> w_chunk((size_t)group_size);

    for (int k = 0; k < K; ++k) {
        const std::uint32_t* q_row_words = q_packed + (size_t)k * W;

        for (int g = 0; g < G; ++g) {
            const int j0 = g * group_size;
            const float scale = scales[k * G + g];
            const float bias  = biases[k * G + g];

            // Initialize bit cursor to the start of this group (j0)
            int bit_pos = j0 * bits;
            int word_idx = bit_pos / 32;
            int bit_off  = bit_pos % 32;

            // Dequantize group_size codes into w_chunk
            for (int t = 0; t < group_size; ++t) {
                std::uint32_t code = unpack_next_code(
                    q_row_words, W, bits, mask, word_idx, bit_off
                );
                // optional safety check (usually removable)
                check(code <= mask, "packed q: code exceeds (2^bits - 1)");
                w_chunk[t] = (float)code * scale + bias;
            }

            // Accumulate dot products
            for (int m = 0; m < M; ++m) {
                const float* a_row = a + (size_t)m * N + j0;
                float sum = 0.0f;
                for (int t = 0; t < group_size; ++t) {
                    sum += a_row[t] * w_chunk[t];
                }
                out[(size_t)m * K + k] += sum;
            }
        }
    }
    return out;
}

/* 
所以我是dequantize a group at a time对吧？然后mlx的implementation是只dequantize一个element at a time?
Yes
补充一点：MLX 虽然概念上是 element-at-a-time，但实现上是“word-at-a-time（8 个 nibble）”地读 packed 权重，
只是每个 nibble 仍然是“解出来马上用”，不落地成 float 缓冲。
https://chatgpt.com/c/693f011e-4d34-8320-b356-707bc4d961d6

它是一个一dequantize还是8个一做？
在那段 MLX 代码里：
读取/解包（unpack）是按 1 个 uint32 一次做的；因为每次循环先拿一个 b_val = b_ptr[b_loc]。
但反量化（dequantize）和乘加（MAC）是按 1 个元素一做的：它在内层 for (pack_idx ...) 里，每次只取出一个 4-bit nibble，立刻 *scale + bias，立刻乘上一个 a 累加。
更直白地说：
一口气拿到 8 个 codes 的“容器”（一个 uint32 里有 8 个 4-bit）。
然后逐个（8 次）：
decode 1 个 nibble
dequantize 1 次
multiply-accumulate 1 次
所以它不是“8 个一起 dequantize 成向量再算”，而是“8 个一组读取，但 1 个一个处理”。（编译器可能会尝试自动向量化，但从源码逻辑上看就是逐元素。）
*/