#include <cstdint>

void quantized_matmul_elementwise_packed_cpp(
    float* out,
    const float* a,
    const int32_t* q,
    const float* scales,
    const float* biases,
    int M,
    int K,
    int N,
    int group_size,
    uint8_t bits
) {
    int G = N / group_size;
    int num_params_per_int32 = 32 / bits;
    int packed_width = (N * bits) >> 5; // N / num_params_per_int32

    for (int idx = 0; idx < M * K; ++idx) {
        out[idx] = 0.0f;
    }
    
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            int g = j / group_size;
            
            int packed_col = j / num_params_per_int32;  // j / (32 / num_params_per_int32)
            int shift = (j % num_params_per_int32) * bits;
            
            // uint32_t packed_val = static_cast<uint32_t>(q[i * packed_width + packed_col]);
            uint8_t val_int = (q[i * packed_width + packed_col] >> shift) & 0x0F;

            float scale = scales[i * G + g];
            float bias = biases[i * G + g];

            float w_val = static_cast<float>(val_int) * scale + bias;

            for (int m = 0; m < M; ++m) {
                out[m * K + i] += a[m * N + j] * w_val;
            }
        }
    }
}

void quantized_matmul_elementwise_packed_cpp_v2(
    float* out,
    const float* a,
    const int32_t* q,
    const float* scales,
    const float* biases,
    int M,
    int K,
    int N,
    int group_size,
    uint8_t bits
) {
    int G = N / group_size;
    int num_params_per_int32 = 32 / bits;
    int packed_width = (N * bits) >> 5; // N / num_params_per_int32

    for (int idx = 0; idx < M * K; ++idx) {
        out[idx] = 0.0f;
    }

    uint8_t shift = 0;
    for (int i = 0; i < K; ++i) {
        int packed_col = 0 ;
        for (int g = 0; g < G; ++g) {
            float scale = scales[i * G + g];
            float bias = biases[i * G + g];
            int g_offset = g*group_size;
            for (int j = 0; j < group_size; ++j) {
                shift += bits;
                shift %= 32;
                packed_col += bits;

                uint8_t val_int = (q[i * packed_width + packed_col >> 5] >> shift) & 0x0F;
                float w_val = static_cast<float>(val_int) * scale + bias;

                for (int m = 0; m < M; ++m) {
                    out[m * K + i] += a[m * N + j] * w_val;
                }
            }
        }
    }
}

void quantized_matmul_elementwise_cpp(
    float* out,
    const float* a,
    const uint8_t* q,
    const float* scales,
    const float* biases,
    int M,
    int K,
    int N,
    int group_size
) {
    int G = N / group_size;

    for (int idx = 0; idx < M * K; ++idx) {
        out[idx] = 0.0f;
    }

    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            int g = j / group_size;

            float scale = scales[i * G + g];
            float bias = biases[i * G + g];

            float w_val = static_cast<float>(q[i * N + j]) * scale + bias;

            for (int m = 0; m < M; ++m) {
                out[m * K + i] += a[m * N + j] * w_val;
            }
        }
    }
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