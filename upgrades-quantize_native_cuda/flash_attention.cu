/*
  double *laplacian_d = NULL;
  cudaMalloc(&laplacian_d, (size_t)Nx * (size_t)Ny * sizeof(double));
  cudaMemset(laplacian_d, 0, (size_t)Nx * (size_t)Ny * sizeof(double));

  cudaMemcpy(laplacian, laplacian_d,
             (size_t)Nx * (size_t)Ny * sizeof(double),
             cudaMemcpyDeviceToHost);
*/

#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void flash_attention_kernel_v2(
    const float* Q, const float* K, const float* V,
    const int L, const int d,
    const int Tc, const int Tr,
    const float softmax_scale,
    float* O
) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;

    int qkv_base_offset = (by * gridDim.x * L * d) + (bx * L * d);

    extern __shared__ float sram[];
    int tile_size = WARP_SIZE * d;

    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S  = &sram[tile_size * 3];

    for (int i = 0; i < Tr; i++) {
        for (int x = 0; x < d; x++) {
            Qi[(tx * d) + x] = Q[qkv_base_offset + (tile_size * i) + (tx * d) + x];
        }

        float m_curr = -INFINITY;
        float l_curr = 0.0f;

        for (int x = 0; x < d; x++) {
            O[qkv_base_offset + (tile_size * i) + (tx * d) + x] = 0.0f;
        }

        __syncthreads();

        for (int j = 0; j < Tc; j++) {
            for (int x = 0; x < d; x++) {
                Kj[(tx * d) + x] = K[qkv_base_offset + (tile_size * j) + (tx * d) + x];
                Vj[(tx * d) + x] = V[qkv_base_offset + (tile_size * j) + (tx * d) + x];
            }
            __syncthreads();

            float row_m_block = -INFINITY;
            for (int y = 0; y < WARP_SIZE; y++) {
                float sum = 0.0f;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(WARP_SIZE * tx) + y] = sum;
                row_m_block = fmaxf(row_m_block, sum);
            }

            float row_l_block = 0.0f;
            for (int y = 0; y < WARP_SIZE; y++) {
                float p = __expf(S[(WARP_SIZE * tx) + y] - row_m_block);
                S[(WARP_SIZE * tx) + y] = p;
                row_l_block += p;
            }

            float m_new = fmaxf(m_curr, row_m_block);
            float alpha = (m_curr == -INFINITY) ? 0.0f : __expf(m_curr - m_new);
            float beta  = __expf(row_m_block - m_new);

            float l_new = (alpha * l_curr) + (beta * row_l_block);
            float inv_l_new = 1.0f / l_new;

            for (int x = 0; x < d; x++) {
                float pv = 0.0f;
                for (int y = 0; y < WARP_SIZE; y++) {
                    pv += S[(WARP_SIZE * tx) + y] * Vj[(y * d) + x];
                }

                int out_idx = qkv_base_offset + (tile_size * i) + (tx * d) + x;
                float o_prev = O[out_idx];

                float numerator = (alpha * l_curr * o_prev) + (beta * pv);
                O[out_idx] = numerator * inv_l_new;
            }

            m_curr = m_new;
            l_curr = l_new;

            __syncthreads();
        }

    }
}

torch::Tensor forward_v2(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    const int B  = Q.size(0);
    const int nh = Q.size(1);
    const int L  = Q.size(2);
    const int d  = Q.size(3);

    const int Bc = WARP_SIZE;
    const int Br = WARP_SIZE;

    const int Tc = (L + Bc - 1) / Bc;
    const int Tr = (L + Br - 1) / Br;
    const float softmax_scale = 1.0f / sqrtf((float)d);

    auto O = torch::zeros_like(Q);

    const int sram_size =
        (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));

    dim3 grid_dim(B, nh);
    dim3 block_dim(WARP_SIZE);

    flash_attention_kernel_v2<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        L, d, Tc, Tr, softmax_scale,
        O.data_ptr<float>()
    );

    return O;
}

torch::Tensor forward_v2(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    const int B  = Q.size(0);
    const int nh = Q.size(1);
    const int L  = Q.size(2);
    const int d  = Q.size(3);

    const int Bc = WARP_SIZE;
    const int Br = WARP_SIZE;

    const int Tc = (L + Bc - 1) / Bc;
    const int Tr = (L + Br - 1) / Br;
    const float softmax_scale = 1.0f / sqrtf((float)d);

    auto O = torch::zeros_like(Q);

    const int sram_size =
        (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));

    dim3 grid_dim(B, nh);
    dim3 block_dim(WARP_SIZE);

    flash_attention_kernel_v2<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        L, d, Tc, Tr, softmax_scale,
        O.data_ptr<float>()
    );

    return O;
}

__global__ void flash_attention_kernel(const float* Q, const float* K, const float* V,
                                     const int L, const int d,
                                     const int Tc, const int Tr, const int Bc, const int Br,
                                     const float softmax_scale,
                                     float* l, float* m, float* O) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;

    int qkv_offset = (by * gridDim.x * L * d) + (bx * L * d);
    int lm_offset  = (by * gridDim.x * L) + (bx * L);   

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S  = &sram[tile_size * 3];

    for (int x = 0; x < L; x += WARP_SIZE) { // Similar to Memory coalescing
        if (x + tx < L) {
            m[lm_offset + tx + x] = -INFINITY;
            l[lm_offset + tx + x] = 0;
        }
    }

    for (int j = 0; j < Tc; j++) {
        // Load to shared memory
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        __syncthreads();  // such that the inner loop can use the correct Kj, Vj

        for (int i = 0; i < Tr; i++)  {

            // Load Qi to SRAM, l and m to registers
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            }
            float row_m_prev = m[lm_offset + tx];
            float row_l_prev = l[lm_offset + tx];

            // S = QK^T, row_m = rowmax(S)
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;

                row_m = max(row_m, sum);
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            // Compute new m and l
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) +
                              (__expf(row_m - row_m_new) * row_l);

            // Write O, l, m to HBM
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1.0f / row_l_new) *
                    ((row_l_prev * __expf(row_m_prev - row_m_new) *
                      O[qkv_offset + (tile_size * i) + (tx * d) + x]) +
                     (__expf(row_m - row_m_new) * pv));
            }

            m[lm_offset + tx] = row_m_new;
            l[lm_offset + tx] = row_l_new;
        }
        __syncthreads();  // prevent thread from using the wrong Kj, Vj in inner loop
    }
}

// https://chatgpt.com/share/693eeb9e-75fc-8003-a6fb-fff199dd52aa
torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, bool use_tensor_cores) {
    int Bc = WARP_SIZE, Br = WARP_SIZE;

    const int B  = Q.size(0);
    const int nh = Q.size(1);
    const int L  = Q.size(2);
    const int d  = Q.size(3);

    const int Tc = (L + Bc - 1) / Bc;
    const int Tr = (L + Br - 1) / Br;
    const float softmax_scale = 1.0f / sqrtf((float)d);

    auto O = torch::zeros_like(Q);

    float* l; float* m;
    cudaMalloc((void**)&l, B * nh * L * sizeof(float));
    cudaMalloc((void**)&m, B * nh * L * sizeof(float));

    const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));

    dim3 grid_dim(B, nh);
    dim3 block_dim(WARP_SIZE);

    flash_attention_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        L, d, Tc, Tr, Bc, Br, softmax_scale,
        l, m, O.data_ptr<float>()
    );

    cudaFree(l);
    cudaFree(m);

    return O;
}

__global__ void flash_attention_kernel_v3(
    const float* Q, const float* K, const float* V,
    const int L, const int d,
    const int Tc, const int Tr,
    const float softmax_scale,
    float* O
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int qkv_base_offset = (by * gridDim.x * L * d) + (bx * L * d);

    extern __shared__ float sram[];
    int tile_size = WARP_SIZE * d;

    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S  = &sram[tile_size * 3];

    float* R = &S[WARP_SIZE * WARP_SIZE];
    float* row_mbuf = &R[WARP_SIZE * WARP_SIZE];
    float* row_lbuf = &row_mbuf[WARP_SIZE];
    float* row_abuf = &row_lbuf[WARP_SIZE];
    float* row_bbuf = &row_abuf[WARP_SIZE];
    float* row_mnew = &row_bbuf[WARP_SIZE];
    float* row_lnew = &row_mnew[WARP_SIZE];

    for (int i = 0; i < Tr; i++) {
        int q_row = i * WARP_SIZE + ty;
        int q_off = qkv_base_offset + q_row * d;

        for (int x = tx; x < d; x += WARP_SIZE) {
            Qi[ty * d + x] = Q[q_off + x];
            O[q_off + x] = 0.0f;
        }

        float m_curr = -INFINITY;
        float l_curr = 0.0f;

        __syncthreads();

        for (int j = 0; j < Tc; j++) {
            int k_row = j * WARP_SIZE + ty;
            int k_off = qkv_base_offset + k_row * d;

            for (int x = tx; x < d; x += WARP_SIZE) {
                Kj[ty * d + x] = K[k_off + x];
                Vj[ty * d + x] = V[k_off + x];
            }

            __syncthreads();

            float row_m_block = -INFINITY;

            for (int y = 0; y < WARP_SIZE; y++) {
                float partial = 0.0f;
                for (int x = tx; x < d; x += WARP_SIZE) {
                    partial += Qi[ty * d + x] * Kj[y * d + x];
                }

                int base = ty * WARP_SIZE + tx;
                R[base] = partial;
                __syncthreads();

                for (int stride = WARP_SIZE / 2; stride > 0; stride >>= 1) {
                    if (tx < stride) {
                        R[ty * WARP_SIZE + tx] += R[ty * WARP_SIZE + tx + stride];
                    }
                    __syncthreads();
                }

                if (tx == 0) {
                    float s = R[ty * WARP_SIZE + 0] * softmax_scale;
                    S[ty * WARP_SIZE + y] = s;
                    row_m_block = fmaxf(row_m_block, s);
                }

                __syncthreads();
            }

            if (tx == 0) {
                row_mbuf[ty] = row_m_block;
            }
            __syncthreads();

            row_m_block = row_mbuf[ty];

            float p = __expf(S[ty * WARP_SIZE + tx] - row_m_block);
            S[ty * WARP_SIZE + tx] = p;

            R[ty * WARP_SIZE + tx] = p;
            __syncthreads();

            for (int stride = WARP_SIZE / 2; stride > 0; stride >>= 1) {
                if (tx < stride) {
                    R[ty * WARP_SIZE + tx] += R[ty * WARP_SIZE + tx + stride];
                }
                __syncthreads();
            }

            float row_l_block = 0.0f;
            if (tx == 0) {
                row_l_block = R[ty * WARP_SIZE + 0];
                row_lbuf[ty] = row_l_block;

                float m_new = fmaxf(m_curr, row_m_block);
                float alpha = (m_curr == -INFINITY) ? 0.0f : __expf(m_curr - m_new);
                float beta  = __expf(row_m_block - m_new);
                float l_new = (alpha * l_curr) + (beta * row_l_block);

                row_abuf[ty] = alpha;
                row_bbuf[ty] = beta;
                row_mnew[ty] = m_new;
                row_lnew[ty] = l_new;
            }
            __syncthreads();

            float alpha = row_abuf[ty];
            float beta  = row_bbuf[ty];
            float m_new = row_mnew[ty];
            float l_new = row_lnew[ty];
            float inv_l_new = 1.0f / l_new;

            for (int x = tx; x < d; x += WARP_SIZE) {
                float pv = 0.0f;
                for (int y = 0; y < WARP_SIZE; y++) {
                    pv += S[ty * WARP_SIZE + y] * Vj[y * d + x];
                }

                float o_prev = O[q_off + x];
                float numerator = (alpha * l_curr * o_prev) + (beta * pv);
                O[q_off + x] = numerator * inv_l_new;
            }

            m_curr = m_new;
            l_curr = l_new;

            __syncthreads();
        }

        __syncthreads();
    }
}

torch::Tensor forward_v3(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    const int B  = Q.size(0);
    const int nh = Q.size(1);
    const int L  = Q.size(2);
    const int d  = Q.size(3);

    const int Bc = WARP_SIZE;
    const int Br = WARP_SIZE;

    const int Tc = (L + Bc - 1) / Bc;
    const int Tr = (L + Br - 1) / Br;
    const float softmax_scale = 1.0f / sqrtf((float)d);

    auto O = torch::zeros_like(Q);

    const int extra_floats = (Bc * Br) + (6 * Br);
    const int sram_size =
        (3 * Bc * d * sizeof(float)) +
        (Bc * Br * sizeof(float)) +
        (extra_floats * sizeof(float));

    dim3 grid_dim(B, nh);
    dim3 block_dim(WARP_SIZE, WARP_SIZE);

    flash_attention_kernel_v3<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        L, d, Tc, Tr, softmax_scale,
        O.data_ptr<float>()
    );

    return O;
}

// Streaming processor (CUDA Core)
// Cores in an SM are grouped further into processing blocks.
// 1 Processing Block = 32 Cores = 1 Warp or it can have multiple warps

/* 
共容/互斥：__host__ 和 __device__ 可同时出现（生成 host+device 两个版本）；__global__ 与 __host__/__device__ 互斥。
调用关系：host 不能调用 __device__；host 可以 <<<>>> 调 __global__；
kernel 内可普通调用 __device__/__host__ __device__(device 版)。
kernel 内不能把另一个 __global__ 当普通函数调用；只能用 child<<<>>> 做 device-side launch
（需 Dynamic Parallelism + 相应编译/链接选项），并且 launch 有额外开销/同步语义。
*/

/*
 * - CPU: optimized for sequential performance
 *   - Large chip area for fast arithmetic (strong single-thread cores)
 *   - Multi-level caches for low-latency data access
 *   - Branch prediction, out-of-order execution, etc.
 *
 * - GPU: optimized for massive parallel throughput
 *   - Very large number of cores/ALUs to run many threads concurrently
 *   - Higher memory bandwidth (to feed many threads)
 *
 * - Host-side sync: after a kernel launch, call cudaDeviceSynchronize()
 *   to block the CPU until the kernel finishes (and to surface async errors).
 *
 * - Block-level sync: __syncthreads() is an intrinsic barrier that
 *   synchronizes all threads within the *same* thread block.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_DIM_X 8
#define BLOCK_DIM_Y 8

__device__ __forceinline__ double func(double x, double y)
{
  return x*x + y*y;
}

__global__ void compute_laplacian(int Nx, int Ny,
    double xstart, double ystart, double dx, double dy, double *result)
{
  __shared__ double f[BLOCK_DIM_X][BLOCK_DIM_Y];

  int i = threadIdx.x, j = threadIdx.y;
  int nx = BLOCK_DIM_X - 2, ny = BLOCK_DIM_Y - 2;

  int ix = (int)blockIdx.x * nx + i - 1;
  int iy = (int)blockIdx.y * ny + j - 1;

  double v = 0.0;
  if (0 <= ix && ix < Nx && 0 <= iy && iy < Ny) {
    double x = xstart + ix * dx;
    double y = ystart + iy * dy;
    v = func(x, y);
  }
  f[i][j] = v;
  __syncthreads();

  if (1 <= i && i <= nx && 1 <= j && j <= ny &&
      1 <= ix && ix < Nx-1 && 1 <= iy && iy < Ny-1) {
    result[ix * Ny + iy] =
        (f[i-1][j] - 2.0*f[i][j] + f[i+1][j]) / (dx*dx) +
        (f[i][j-1] - 2.0*f[i][j] + f[i][j+1]) / (dy*dy);
  }
}

int main()
{
  double xstart = -1.0, ystart = -1.0;
  int Nx = 200, Ny = 400;

  if (Nx < 3 || Ny < 3) return 1;

  double dx = 2.0 / (Nx - 1);
  double dy = 2.0 / (Ny - 1);

  double *laplacian_d = NULL;
  cudaMalloc(&laplacian_d, (size_t)Nx * (size_t)Ny * sizeof(double));
  cudaMemset(laplacian_d, 0, (size_t)Nx * (size_t)Ny * sizeof(double));

  dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
  int nx = BLOCK_DIM_X - 2, ny = BLOCK_DIM_Y - 2;
  dim3 dimGrid((Nx + nx - 1) / nx, (Ny + ny - 1) / ny);

  compute_laplacian<<<dimGrid, dimBlock>>>(Nx, Ny, xstart, ystart, dx, dy, laplacian_d);
  cudaDeviceSynchronize();

  double *laplacian = (double*)malloc((size_t)Nx * (size_t)Ny * sizeof(double));
  cudaMemcpy(laplacian, laplacian_d,
             (size_t)Nx * (size_t)Ny * sizeof(double),
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < Nx; i++) {
    for (int j = 0; j < Ny; j++)
      printf("%lf\t", laplacian[i * Ny + j]);
    printf("\n");
  }

  cudaFree(laplacian_d);
  free(laplacian);
  return 0;
}

#define BLOCK_DIM 16
#define n 200
#define COARSENING_FACTOR 4

__global__ void mat_mul(int N, double *A, double *B, double *C)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < N && j < N) {
    double sum = 0;
    for (int k = 0; k < N; k++)
      sum += A[i * N + k] * B[k * N + j];
    C[i * N + j] = sum;
  }
}

__global__ void mat_mul_v2(int N, double *A, double *B, double *C)
{
  __shared__ double As[BLOCK_DIM][BLOCK_DIM];
  __shared__ double Bs[BLOCK_DIM][BLOCK_DIM];
  int i = blockIdx.x * BLOCK_DIM + threadIdx.x;
  int j = blockIdx.y * BLOCK_DIM + threadIdx.y;

  double sum = 0;
  for (int p = 0; p < (N+BLOCK_DIM-1) / BLOCK_DIM; p++) {
    int k_start = p * BLOCK_DIM;

    if (i < N && k_start + threadIdx.y < N)
      As[threadIdx.x][threadIdx.y] =
        A[i * N + k_start + threadIdx.y];

    if (k_start + threadIdx.x < N && j < N)
      Bs[threadIdx.x][threadIdx.y] =
        B[(k_start + threadIdx.x) * N + j];

    __syncthreads();

    if (i < N && j < N)
      for (int k = 0; k < BLOCK_DIM && k_start + k < N; k++)
        sum += As[threadIdx.x][k] * Bs[k][threadIdx.y];

    __syncthreads();
  }

  if (i < N && j < N) C[i * N + j] = sum;
}

__global__ void mat_mul_v3(int N, double *A, double *B, double *C)
{
  __shared__ double As[BLOCK_DIM][BLOCK_DIM], Bs[BLOCK_DIM][BLOCK_DIM];
  int i = blockIdx.x * BLOCK_DIM + threadIdx.x;
  int jstart = (blockIdx.y * BLOCK_DIM + threadIdx.y) * COARSENING_FACTOR;

  double Cvalue[COARSENING_FACTOR];
  for (int c = 0; c < COARSENING_FACTOR; c++) Cvalue[c] = 0;
  
  for (int p = 0; p < (N+BLOCK_DIM-1) / BLOCK_DIM; p++) {
    int k_start = p * BLOCK_DIM;

    if (i < N && k_start + threadIdx.y < N)
      As[threadIdx.x][threadIdx.y] = A[i * N + k_start + threadIdx.y];

    for (int c = 0; c < COARSENING_FACTOR; c++) {
      int j = jstart + c;
      if (k_start + threadIdx.x < N && j < N)
        Bs[threadIdx.x][threadIdx.y] = B[(k_start + threadIdx.x) * N + j];
      __syncthreads();

      if (i < N && j < N)
        for (int k = 0; k < BLOCK_DIM && k_start + k < N; k++)
          Cvalue[c] += As[threadIdx.x][k] * Bs[k][threadIdx.y];
      __syncthreads();
    }
  }

  for (c = 0; c < COARSENING_FACTOR; c++) {
    int j = jstart + c;
    if (i < N && j < N) C[i * N + j] = Cvalue[c];
  }
}

/* 
Each time a location in the global memory is accessed, a range of
consecutive locations that include the requested location can be
transferred to the processor at high speed. This is known as DRAM
bursts. Adjacent memory for multiple threads.
*/

__global__ void mat_mul_v4(int N, double *A, double *B, double *C)
{
  __shared__ double As[BLOCK_DIM][BLOCK_DIM], Bs[BLOCK_DIM][BLOCK_DIM];
  int i = blockIdx.x * BLOCK_DIM + threadIdx.x, c;
  int j = blockIdx.y * BLOCK_DIM * COARSENING_FACTOR + threadIdx.y;

  double Cvalue[COARSENING_FACTOR];
  for (c = 0; c < COARSENING_FACTOR; c++) Cvalue[c] = 0;
  
  for (int p = 0; p < (N+BLOCK_DIM-1) / BLOCK_DIM; p++) {
    int k_start = p * BLOCK_DIM;

    if (i < N && k_start + threadIdx.y < N)
      As[threadIdx.x][threadIdx.y] = A[i * N + k_start + threadIdx.y];

    for (c = 0; c < COARSENING_FACTOR; c++) {
      int real_j = j + c * BLOCK_DIM;
      if (k_start + threadIdx.x < N && real_j < N)
        Bs[threadIdx.x][threadIdx.y] = B[(k_start + threadIdx.x) * N + real_j];
      __syncthreads();

      if (i < N && real_j < N)
        for (int k = 0; k < BLOCK_DIM && k_start + k < N; k++)
          Cvalue[c] += As[threadIdx.x][k] * Bs[k][threadIdx.y];
      __syncthreads();
    }
  }

  for (c = 0; c < COARSENING_FACTOR; c++) {
    int real_j = j + c * BLOCK_DIM;
    if (i < N && real_j < N) C[i * N + real_j] = Cvalue[c];
  }
}

/*Thread coarsening:
I We will have less data accesses from the global memory if the
number of blocks is smaller, or the block dimension is larger.
I However, the block dimension is limited by the shared memory.
I When the size of the matrices is large, the number of blocks may be
many times the number of SMs, so that some blocks are actually
executed sequentially.
I In this case, we may increase the workload of each thread and
decrease the number of blocks, which often achieves better
ciency. Technique is called thread coarsening. */

void mat_mul_main() {
  double *A, *B, *C, *A_d, *B_d, *C_d;
  int i;

  cudaMalloc(&A_d, n * n * sizeof(double));
  cudaMalloc(&B_d, n * n * sizeof(double));
  cudaMalloc(&C_d, n * n * sizeof(double));

  A = (double *)malloc(sizeof(double) * n * n);
  B = (double *)malloc(sizeof(double) * n * n);
  C = (double *)malloc(sizeof(double) * n * n);

  for (i = 0; i < n * n; i++) {
    A[i] = 1. * rand() / RAND_MAX;
    B[i] = 1. * rand() / RAND_MAX;
  }

  cudaMemcpy(A_d, A, n * n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, n * n * sizeof(double), cudaMemcpyHostToDevice);

  dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
  dim3 dimGrid((n+BLOCK_DIM-1) / BLOCK_DIM, (n+BLOCK_DIM-1) / (BLOCK_DIM*COARSENING_FACTOR));
  mat_mul<<<dimGrid, dimBlock>>>(n, A_d, B_d, C_d);
  cudaDeviceSynchronize();

  cudaMemcpy(C, C_d, n * n * sizeof(double), cudaMemcpyDeviceToHost);
  printf("Maximum difference: %f\n", max_diff(n, A, B, C));

  cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
  free(A); free(B); free(C);
  return 0;
}

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void all_sum_reduce(int *in, int *out, int n) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    
    sdata[tid] = (tid < n) ? in[tid] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) *out = sdata[0];
}

int main() {
    int n = 256;
    int *h_in = (int*)malloc(size);
    int h_out;
    int *d_in, *d_out;

    for (int i = 0; i < n; i++) h_in[i] = 1;

    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, sizeof(int));

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    int threads = 1;
    while (threads < n) threads *= 2;
    int size = n * sizeof(int);

    all_sum_reduce<<<1, threads, threads*sizeof(int)>>>(d_in, d_out, n);

    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Sum: %d\n", h_out);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);

    return 0;
}

// 树状数组lowbit是管理的长度 (i-lowbit(i), i]
// 加的时候只要考虑后续有哪些节点管理的区间包含当前节点

__global__ void segmented_scan_native(int n, int *array, int *prefix_sum)
{
    __shared__ int arr[BLOCK_DIM], location, mutex;
    int leave = 0, idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[threadIdx.x] = array[idx];
    if (threadIdx.x == 0) location = mutex = 0;
    __syncthreads();

    if (threadIdx.x != 0 && idx < n)
        while (!leave) {
            if (atomicExch(&mutex, 1) == 0) {
                location++;
                arr[location] += arr[location-1];
                __threadfence_block();
                leave = 1;
                atomicExch(&mutex, 0);
            }
        }
    __syncthreads();
    if (idx < n) prefix_sum[idx] = arr[threadIdx.x];
}

__global__ void prefix_scan_self_invented(int n, int* array, int *prefix_sum) {
  __shared__ int sdata[];
  
  int tid = threadIdx.x;
  
  sdata[tid] = array[tid];
  __syncthreads();

  for (int stride = n / 2; stride > 0; stride /= 2) {
      
      int val = sdata[tid];
      __syncthreads();

      if (tid + stride < n) {
          sdata[tid + stride] += val;
      }
      
      __syncthreads();
  }

  __syncthreads();
  if (idx < n) prefix_sum[idx] = array[threadIdx.x];
}

__global__ void prefix_scan_koggle_stone(int n, int *array, int *prefix_sum)
{
    __shared__ int arr[BLOCK_DIM];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) arr[threadIdx.x] = array[idx];
    for (int stride = 1; stride < BLOCK_DIM; stride *= 2)
    {
        __syncthreads();
        if (threadIdx.x >= stride)
            arr[threadIdx.x] += arr[threadIdx.x - stride];
    }

    __syncthreads();
    if (idx < n) prefix_sum[idx] = arr[threadIdx.x];
}

__device__ int block_idx = 0;
__device__ int block_counter = 0;

__device__ void prefix_scan_koggle_stone(int n, int *array, int *prefix_sum)
{
    __shared__ int arr[BLOCK_DIM];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) arr[threadIdx.x] = array[idx];
    for (int stride = 1; stride < BLOCK_DIM; stride *= 2)
    {
        __syncthreads();
        if (threadIdx.x >= stride)
            arr[threadIdx.x] += arr[threadIdx.x - stride];
    }

    __syncthreads();
    if (idx < n) prefix_sum[idx] = arr[threadIdx.x];
}

__global__ void prefix_scan_brent_kung_native(int n, int *array, int *prefix_sum)
{
    __shared__ int arr[BLOCK_DIM];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) arr[threadIdx.x] = array[idx];
    
    for (int stride = 1; stride < BLOCK_DIM; stride *= 2) {
        __syncthreads();
        if ((threadIdx.x + 1) % (stride * 2) == 0)
            arr[threadIdx.x] += arr[threadIdx.x - stride];
    }

    for (int stride = BLOCK_DIM/4; stride > 0; stride /= 2) {
        __syncthreads();
        if (threadIdx.x + 1 > stride && 
           (threadIdx.x + 1 - stride) % (stride * 2) == 0)
        {
            arr[threadIdx.x] += arr[threadIdx.x - stride];
        }
    }

    __syncthreads();
    if (idx < n) prefix_sum[idx] = arr[threadIdx.x];
}

__global__ void prefix_scan_brent_kung_optimzed_for_ctrl_divergence(int n, int *array, int *prefix_sum)
{
    __shared__ int arr[BLOCK_DIM];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) arr[threadIdx.x] = array[idx];
    
    for (int stride = 1; stride < BLOCK_DIM; stride *= 2) {
        __syncthreads();
        if (threadIdx < BLOCK_DIM / (stride * 2)) {
          int mockIdx = threadIdx.x * stride;
          arr[mockIdx+stide] += arr[mockIdx];
        }
    }

    for (int stride = BLOCK_DIM/4; stride > 0; stride /= 2) {
        __syncthreads();
        if (threadIdx.x < BLOCK_DIM / (stride * 2)) {
          int mockIdx = (threadIdx.x+1) * stride * 2;
          arr[mockIdx + stride] += arr[mockIdx];
        }
    }

    __syncthreads();
    if (idx < n) prefix_sum[idx] = arr[threadIdx.x];
}

/*
__global__ void prefix_scan_brent_kung_optimzed_for_ctrl_divergence(int n, int *array, int *prefix_sum)
{
    __shared__ int arr[2*BLOCK_DIM];
    int idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;

    // Load 2 elements per thread into shared memory
    if (idx < n) arr[threadIdx.x] = array[idx];
    if (idx + BLOCK_DIM < n) arr[threadIdx.x + BLOCK_DIM] = array[idx + BLOCK_DIM];

    // Reduction Phase (Up-Sweep)
    for (int stride = 1; stride <= BLOCK_DIM; stride *= 2) {
        __syncthreads();
        int i = stride * 2 * (threadIdx.x + 1) - 1;
        if (i < 2*BLOCK_DIM) arr[i] += arr[i - stride];
    }

    // Distribution Phase (Down-Sweep)
    for (int stride = BLOCK_DIM/2; stride > 0; stride /= 2) {
        __syncthreads();
        int i = 2 * stride * threadIdx.x - 1 + stride;
        if (threadIdx.x > 0 && i < 2*BLOCK_DIM) arr[i] += arr[i - stride];
    }

    __syncthreads();
    
    // Write results back to global memory
    if (idx < n) prefix_sum[idx] = arr[threadIdx.x];
    if (idx + BLOCK_DIM < n) prefix_sum[idx + BLOCK_DIM] = arr[threadIdx.x + BLOCK_DIM];
}*/

__global__ void SinglePass(int n, int *array, int *partial_sum, int *prefix_sum)
{
    __shared__ int arr[2*BLOCK_DIM], scan_block_idx;
    
    if (threadIdx.x == 0)
        scan_block_idx = atomicAdd(&block_idx, 1);
    __syncthreads();

    int idx = 2 * scan_block_idx * blockDim.x + threadIdx.x;

    if (idx < n) arr[threadIdx.x] = array[idx];
    if (idx + BLOCK_DIM < n) arr[threadIdx.x + BLOCK_DIM] = array[idx + BLOCK_DIM];
    
    prefix_scan_koggle_stone(BLOCK_DIM, arr, arr+BLOCK_DIM);

    __syncthreads();

    if (threadIdx.x == 0) {
        if (scan_block_idx != 0)
            while (atomicAdd(&block_counter, 0) != scan_block_idx) {}
        else
            partial_sum[0] = 0;

        partial_sum[scan_block_idx + 1] = partial_sum[scan_block_idx] + arr[2*BLOCK_DIM-1];
        
        __threadfence();
        atomicAdd(&block_counter, 1);
    }
    __syncthreads();

    if (idx < n)
        prefix_sum[idx] = arr[threadIdx.x] + partial_sum[scan_block_idx];
    if (idx + BLOCK_DIM < n)
        prefix_sum[idx + BLOCK_DIM] = arr[threadIdx.x + BLOCK_DIM] + partial_sum[scan_block_idx];

    __syncthreads();
}

int prefix_scan_main()
{
    int *array = (int *)malloc(N * sizeof(int));
    int *prefix_sum = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++)
        array[i] = i;

    int *array_d, *prefix_sum_d;
    cudaMalloc(&array_d, N * sizeof(int));
    cudaMalloc(&prefix_sum_d, N * sizeof(int));
    cudaMemcpy(array_d, array, N * sizeof(int), cudaMemcpyHostToDevice);

    segmented_scan<<<(N-1) / BLOCK_DIM + 1, BLOCK_DIM>>>(N, array_d, prefix_sum_d);
    cudaDeviceSynchronize();

    cudaMemcpy(prefix_sum, prefix_sum_d, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(array_d);
    cudaFree(prefix_sum_d);
    free(array);

    for (int i = 0; i < N; i++)
        printf("%d\n", prefix_sum[i]);
    free(prefix_sum);
    return 0;
}

/*
Radix sort
destination of a zero = # zeros before = key index - # ones before
destination of a one = # zeros in total + # ones before = 
# keys in total - # ones in total + ones before.
*/

__global__ void radix_sort_iter(unsigned int iter,
    unsigned int *input, unsigned int *output,
    unsigned int *bits, unsigned int N)
{
    unsigned int bit, idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        bits[idx+1] = bit = (input[idx] >> iter) & 1;
    __syncthreads();

    parallel_scan(bits + 1, N);

    if (idx < N) {
        unsigned int dst = bit ? N - bits[N] + bits[idx]
                               : idx - bits[idx];

        output[dst] = input[idx];
    }
}

/*
Assume there are K blocks. Then
(destination of the first zero in block k) = \sum_{j=0}^{k-1} (# zeros in block j),
(destination of the first one in block k) = \sum_{j=0}^{K-1} (# zeros in block j)
                                           + \sum_{j=0}^{k-1} (# ones in block k).
*/

// Further shared memory optimization: https://gemini.google.com/app/76e87e922ebb948f