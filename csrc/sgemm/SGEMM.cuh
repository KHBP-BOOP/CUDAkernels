#include <cuda_runtime_api.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>


namespace {

    __device__ float4& FLOAT4(float& f) {

        return reinterpret_cast<float4&>(f);
    }

    __device__ const float4& CONST_FLOAT4(const float& f) {

        return reinterpret_cast<const float4&>(f);
    }

    // col & 3 保持不变，故col为4的倍数时结果仍为4的倍数（STS.128 的 16B 对齐不变）
    __device__ __forceinline__ int swizzle_col_mask(int mask, int col) {
        return (((col >> 2) ^ mask) << 2) | (col & 3);
    }

    // CHUNKS - 每行中float4类型数据的个数，必须为2的幂
    template <int CHUNKS>
    __device__ __forceinline__ int swizzle_col(int row, int col) {
        static_assert((CHUNKS & (CHUNKS - 1)) == 0, "CHUNKS 必须是 2 的幂");
        return swizzle_col_mask(row & (CHUNKS - 1), col);
    }

}

namespace CUDAkernels {


// v2
// 协作加载 tileA: 全局内存 → Shared Memory
// r0 = blockIdx.y * BM,  k = 当前 K 维度起点
template <int BM, int BK, int BLOCK_SIZE>
__device__ void store_tile_A(const float *__restrict__ A, float As[BM][BK],
                            int M, int K, int r0, int k0, int tid, cuda::pipeline<cuda::thread_scope_thread>& pipe)
{
    // 线程重排: 16 * 4 布局, 实现合并访问以及float4向量化 (coalesced access)
    constexpr int A_BLOCK_X = BK / 4; // = 4
    constexpr int A_BLOCK_Y = BLOCK_SIZE / A_BLOCK_X; // = 32
    constexpr int A_TILES = (BM + A_BLOCK_Y - 1) / A_BLOCK_Y; // = 8

    // float4
    // 计算线程块尺寸时应采用向上取整的除法，故存在线程数量大于数据总量的情况
    if (K % 4 == 0) {

        int a_thread_x = tid % A_BLOCK_X;             // 0 ~ 3
        int a_thread_y = tid / A_BLOCK_X;             // 0 ~ 31
        int a_thread_x_float4 = a_thread_x * 4;
        int c = k0 + a_thread_x_float4;

        #pragma unroll
        for (int t = 0; t < A_TILES; ++t) { // 8次loop

            int i = a_thread_y + t * A_BLOCK_Y;
            int r = r0 + i;
            int col_swizzle = swizzle_col<BK / 4>(i, a_thread_x_float4);

            if (r < M && c + 3 < K) {

                cuda::memcpy_async(
                    &FLOAT4(As[i][col_swizzle]),
                    &CONST_FLOAT4(A[r * K + c]),
                    cuda::aligned_size_t<16>(sizeof(float4)),
                    pipe
                );
            }

            else {
                FLOAT4(As[i][col_swizzle]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
    }

    // 标量格式
    else {

        // 扁平索引确保每个线程都覆盖 tile，且不依赖 K 能否被 4 整除
        #pragma unroll
        for (int idx = tid; idx < BM * BK; idx += BLOCK_SIZE) {

            int i = idx / BK;
            int j = idx % BK;
            int r = r0 + i;
            int c = k0 + j;

            As[i][swizzle_col<BK / 4>(i, j)] =
                (r < M && c < K) ? A[r * K + c] : 0.0f;
        }
    }
}

// v2
// 协作加载 tileB: 全局内存 → Shared Memory
// c0 = blockIdx.x * BN,  k = 当前 K 维度起点
template <int BN, int BK, int BLOCK_SIZE>
__device__ void store_tile_B(const float *__restrict__ B, float Bs[BK][BN],
                            int K, int N, int c0, int k0, int tid, cuda::pipeline<cuda::thread_scope_thread>& pipe)
{
    // Bs   32 * 64
    // 线程重排: 16 * 16 布局, 实现合并访问
    constexpr int B_BLOCK_X = BN / 4; // 16
    constexpr int B_BLOCK_Y = BLOCK_SIZE / B_BLOCK_X; // = 16
    constexpr int B_TILES = (BK + B_BLOCK_Y - 1) / B_BLOCK_Y; // 1

    // float4
    if (N % 4 == 0) {

        int b_thread_x = tid % B_BLOCK_X;             // 0 ~ 15
        int b_thread_y = tid / B_BLOCK_X;             // 0 ~ 15
        int b_thread_x_float4 = b_thread_x * 4;
        int c = c0 + b_thread_x_float4;

        #pragma unroll
        for (int t = 0; t < B_TILES; ++t) {

            int j = b_thread_y + t * B_BLOCK_Y; // 0 ~ 31
            int r = k0 + j;

            if (r < K && c + 3 < N) {

                cuda::memcpy_async(
                    &FLOAT4(Bs[j][b_thread_x_float4]),
                    &CONST_FLOAT4(B[r * N + c]),
                    cuda::aligned_size_t<16>(sizeof(float4)),
                    pipe
                );
            }

            else {

                FLOAT4(Bs[j][b_thread_x_float4]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
    }

    // 标量格式
    else {

        // 扁平索引确保每个线程都覆盖 tile，且不依赖 N 能否被 4 整除
        #pragma unroll
        for (int idx = tid; idx < BK * BN; idx += BLOCK_SIZE) {

            int j = idx / BN;
            int col = idx % BN;
            int r = k0 + j;
            int c = c0 + col;

            Bs[j][col] = (r < K && c < N) ? B[r * N + c] : 0.0f;
        }
    }
}

// v2
// float4方式读取、无边界判断、无零填充版本，仅当M可被BM整除、N可被BN整除、K可被BK整除同时成立时使用
// as_stage / a_src 已含线程偏移，函数内只做常量步进
template <int BM, int BK, int BLOCK_SIZE, int A_TILES, int A_BLOCK_Y>
__device__ __forceinline__ void store_tile_A_full(
    float* __restrict__ As_X_Y_Z, const float* __restrict__ A_X_Y,
    const int A_row_stride,
    cuda::pipeline<cuda::thread_scope_thread>& pipe)
{

    #pragma unroll
    for (int t = 0; t < A_TILES; ++t) {
        cuda::memcpy_async(
            As_X_Y_Z + t * (A_BLOCK_Y * BK),
            A_X_Y + t * A_row_stride,
            cuda::aligned_size_t<16>(sizeof(float4)),   // 必须保留；换成普通 size_t 会退化为 4 条 4B cp.async
            pipe);
    }
}

template <int BN, int BK, int BLOCK_SIZE, int B_TILES, int B_BLOCK_Y>
__device__ __forceinline__ void store_tile_B_full(
    float* __restrict__ Bs_X_Y_Z, const float* __restrict__ B_X_Y,
    const int B_row_stride,
    cuda::pipeline<cuda::thread_scope_thread>& pipe)
{

    #pragma unroll
    for (int t = 0; t < B_TILES; ++t) {

        cuda::memcpy_async(
            Bs_X_Y_Z + t * (B_BLOCK_Y * BN),
            B_X_Y + t * B_row_stride,
            cuda::aligned_size_t<16>(sizeof(float4)),
            pipe);
    }
}




// v2
template <int BM, int BN, int BK, int BLOCK_SIZE, int numStages, bool FULL_TILE>
__global__ void sm89_fp32_gemm(const float* A, const float* B, float* C,
int M, int N, int K) {

    // 16B 对齐：两个 fragment 都按 float4 读取（LDS.128）
    __shared__ __align__(16) float As[numStages][BM][BK]; // 128 16
    __shared__ __align__(16) float Bs[numStages][BK][BN]; // 16  64

    int r0 = blockIdx.y * BM;
    int c0 = blockIdx.x * BN;
    int tid = threadIdx.x;

    // 计算 tileC 、写入C 时的线程排布（16 * 8）
    constexpr int C_BLOCK_X = 8;
    constexpr int C_BLOCK_Y = BLOCK_SIZE / C_BLOCK_X;  // = 16
    int c_thread_x = tid % C_BLOCK_X; // 0 ~ 7
    int c_thread_y = tid / C_BLOCK_X; // 0 ~ 15

    // 16 * 8 threads 负责 128 * 64 个元素
    // 每个线程负责 Tm×Tn 个输出元素
    constexpr int Tm = BM / C_BLOCK_Y;  // = 8 跨步覆盖128行 由BM、BLOCK_SIZE决定
    constexpr int Tn = BN / C_BLOCK_X;  // = 8 跨步覆盖64列 由BN决定

    float Ct[Tm][Tn] = { 0.0f };
    float a_frag[Tm][4] = { 0.0f }; // 执行两轮运输+计算，缩小体积至原先的一半 // 存放float4数据
    float b_frag[Tn] = { 0.0f }; // Tn==8，故优化为连续存储2个float4



    
    constexpr int A_BLOCK_X = BK / 4; // 4
    constexpr int A_BLOCK_Y = BLOCK_SIZE / A_BLOCK_X; // 32
    constexpr int A_TILES   = BM / A_BLOCK_Y; // 2
    // 要求BM能被A_BLOCK_Y整除
    static_assert(BM % A_BLOCK_Y == 0);
    
    constexpr int B_BLOCK_X = BN / 4; // 16
    constexpr int B_BLOCK_Y = BLOCK_SIZE / B_BLOCK_X; // 8
    constexpr int B_TILES   = BK / B_BLOCK_Y; // 2
    // 要求 BK 能被 B_BLOCK_Y 整除
    static_assert(BK % B_BLOCK_Y == 0);


    const int a_thread_x = tid % A_BLOCK_X;
    const int a_thread_y = tid / A_BLOCK_X;
    const int a_thread_x_float4 = a_thread_x * 4;
    const int As_col_swz = swizzle_col<BK / 4>(a_thread_y, a_thread_x_float4);

    // 提前计算出该线程负责计算As中哪个位置的元素（A_TILES大于1时，存在跨步循环现象，会对应多个元素，ta们X轴坐标相同，Y轴坐标等距相隔）
    float* As_X_Y = &As[0][0][0] + a_thread_y * BK + swizzle_col<BK / 4>(a_thread_y, a_thread_x_float4);
    const float* A_X_Y = A + (r0 + a_thread_y) * K + a_thread_x_float4;
    const int A_row_stride = A_BLOCK_Y * K;

    
    const int b_thread_x = tid % B_BLOCK_X;
    const int b_thread_y = tid / B_BLOCK_X;
    const int b_thread_x_float4 = b_thread_x * 4;

    // 同理于 As_X_Y
    float* Bs_X_Y = &Bs[0][0][0] + b_thread_y * BN + b_thread_x_float4;
    const float* B_X_Y = B + b_thread_y * N + c0 + b_thread_x_float4;
    const int B_row_stride = B_BLOCK_Y * N;


    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();


    // 每个线程先异步拷贝*下一轮K循环中计算部分*需要用到的数据，然后执行*本轮K循环*的计算操作

    // pipeline为线程私有
    // 启动前，将前 numStagesBefore 个stage逐个占用，启动异步拷贝，后续以隐藏延迟
    int numStagesBefore = numStages - 1; // 1 待定，但一定小于等于numStages

    for (int stage = 0; stage < numStagesBefore; ++stage) {

        if constexpr (FULL_TILE) {
            store_tile_A_full<BM, BK, BLOCK_SIZE, A_TILES, A_BLOCK_Y>(
                As_X_Y + stage * (BM * BK),
                A_X_Y + stage * BK,
                A_row_stride,
                pipe
            );
            store_tile_B_full<BN, BK, BLOCK_SIZE, B_TILES, B_BLOCK_Y>(
                Bs_X_Y + stage * (BK * BN),
                B_X_Y + stage * (BK * N),
                B_row_stride,
                pipe
            );
        }

        else {

            int k0 = stage * BK; // K方向上的起始地址
            store_tile_A<BM, BK, BLOCK_SIZE>(A, As[stage], M, K, r0, k0, tid, pipe);
            store_tile_B<BN, BK, BLOCK_SIZE>(B, Bs[stage], K, N, c0, k0, tid, pipe);
        }

        pipe.producer_commit();
    }


    // K-Loop
    // 一次循环对应
    for (int k = 0; k < K; k += BK) {

        const int next_k = k + numStagesBefore * BK;

        pipe.producer_acquire();
        if (next_k < K) { // 如果GMEM对应空间中仍有待拷贝的数据

            const int stage = (next_k / BK) % numStages; // 共享内存使用环形 stage

            if constexpr (FULL_TILE) {
                store_tile_A_full<BM, BK, BLOCK_SIZE, A_TILES, A_BLOCK_Y>(
                    As_X_Y + stage * (BM * BK),
                    A_X_Y + next_k,
                    A_row_stride,
                    pipe
                );
                store_tile_B_full<BN, BK, BLOCK_SIZE, B_TILES, B_BLOCK_Y>(
                    Bs_X_Y + stage * (BK * BN),
                    B_X_Y + next_k * N,
                    B_row_stride,
                    pipe
                );
            }
            else {
                store_tile_A<BM, BK, BLOCK_SIZE>(A, As[stage], M, K, r0, next_k, tid, pipe);
                store_tile_B<BN, BK, BLOCK_SIZE>(B, Bs[stage], K, N, c0, next_k, tid, pipe);
            }
        }

        // 包含未prefetch的空组。保证计数对齐
        pipe.producer_commit();


        pipe.consumer_wait();
        __syncthreads();

        const int col0 = c_thread_x * (sizeof(float4) / sizeof(float)); // 0~7  *  8
        const int stage = (k / BK) % numStages;

        #pragma unroll
        for (int p = 0; p < BK; p += 4) {

            // A: 每个输出 row 一条 LDS.128，一次拿到 p=pg..pg+3
            #pragma unroll
            for (int i = 0; i < Tm; ++i) {

                int row = c_thread_y + i * C_BLOCK_Y;

                FLOAT4(*(a_frag[i])) = FLOAT4(As[stage][row][
                swizzle_col<BK / 4>(row, p)
                ]);
            }

            // B: 每个 p 一条 LDS.128
            #pragma unroll
            for (int pp = 0; pp < 4; ++pp) {

                #pragma unroll
                for (int offset = 0; offset < Tn; offset += 4) { // 每个线程的b_frag数组共Tn个位置，存放 Tn / 4 个float4数据

                    FLOAT4(*(b_frag + offset)) = FLOAT4(Bs[stage][p + pp][col0 + offset * C_BLOCK_X]);
                }

                // 每读取完一行Bs，立即计算
                #pragma unroll
                for (int i = 0; i < Tm; ++i) {

                    #pragma unroll
                    for (int j = 0; j < Tn; ++j) {

                        Ct[i][j] += a_frag[i][pp] * b_frag[j];
                    }
                }
            }
        }



        __syncthreads(); //避免在本轮循环计算完成前，SMEM被下一轮数据覆盖
        pipe.consumer_release();
    }


    // 写回结果
    #pragma unroll
    for (int i = 0; i < Tm; i++) {

        int r = r0 + c_thread_y + i * C_BLOCK_Y;
        int c = c0 + c_thread_x * ( sizeof(float4) / sizeof(float) );

        if constexpr (FULL_TILE) {
            #pragma unroll
            for (int offset = 0; offset < Tn; offset += 4) { // 每个线程的b_frag数组共Tn个位置，读取 Tn / 4 个float4数据

                FLOAT4(C[r * N + c + offset * C_BLOCK_X]) = FLOAT4(Ct[i][offset]);
            }
        }
        else if (N % 4 == 0) {

            if (r < M) {

                #pragma unroll
                for (int offset = 0; offset < Tn; offset += 4) {

                    int col = c + offset * C_BLOCK_X;
                    if (col + 3 < N) {
                        FLOAT4(C[r * N + col]) = FLOAT4(Ct[i][offset]);
                    }
                }
            }
        }
        else {

            if (r < M) {

                #pragma unroll
                for (int offset = 0; offset < Tn; ++offset) {

                    int col = c + (offset / 4) * (C_BLOCK_X * 4) + offset % 4;
                    if (col < N) {
                        C[r * N + col] = Ct[i][offset];
                    }
                }
            }
        }
    }
}




// v2
template <int BM, int BN, int BK, int BLOCK_SIZE, int numStages>
__global__ void sm89_fp32_gemm(const float* A, const float* B, float* C,
int M, int N, int K) {

    // 16B 对齐：两个 fragment 都按 float4 读取（LDS.128）
    __shared__ __align__(16) float As[numStages][BM][BK]; // 128 16
    __shared__ __align__(16) float Bs[numStages][BK][BN]; // 16  64

    int r0 = blockIdx.y * BM;
    int c0 = blockIdx.x * BN;
    int tid = threadIdx.x;

    // 计算 tileC 、写入C 时的线程排布（16 * 8）
    constexpr int C_BLOCK_X = 8;
    constexpr int C_BLOCK_Y = BLOCK_SIZE / C_BLOCK_X;  // = 16
    int c_thread_x = tid % C_BLOCK_X; // 0 ~ 7
    int c_thread_y = tid / C_BLOCK_X; // 0 ~ 15

    // 16 * 8 threads 负责 128 * 64 个元素
    // 每个线程负责 Tm×Tn 个输出元素
    constexpr int Tm = BM / C_BLOCK_Y;  // = 8 跨步覆盖128行 由BM、BLOCK_SIZE决定
    constexpr int Tn = BN / C_BLOCK_X;  // = 8 跨步覆盖64列 由BN决定

    float Ct[Tm][Tn] = { 0.0f };
    float a_frag[Tm][4] = { 0.0f }; // 执行两轮运输+计算，缩小体积至原先的一半 // 存放float4数据
    float b_frag[Tn] = { 0.0f }; // Tn==8，故优化为连续存储2个float4



    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();


    // 每个线程先异步拷贝*下一轮K循环中计算部分*需要用到的数据，然后执行*本轮K循环*的计算操作

    // pipeline为线程私有
    // 启动前，将前 numStagesBefore 个stage逐个占用，启动异步拷贝，后续以隐藏延迟
    int numStagesBefore = numStages - 1; // 1 待定，但一定小于等于numStages

    for (int stage = 0; stage < numStagesBefore; ++stage) {

    int k0 = stage * BK; // K方向上的起始地址
    store_tile_A<BM, BK, BLOCK_SIZE>(A, As[stage], M, K, r0, k0, tid, pipe);
    store_tile_B<BN, BK, BLOCK_SIZE>(B, Bs[stage], K, N, c0, k0, tid, pipe);

    pipe.producer_commit();
    }


    // // 保证启动pipeline前的numStages次load全部结束
    // // 避免启动后首次consume时访问到脏数据
    // __syncthreads();


    // K-Loop
    // 一次循环对应
    for (int k = 0; k < K; k += BK) {

    int As_Bs_K_index = k + numStagesBefore * BK;

    pipe.producer_acquire();
    if (As_Bs_K_index < K) { // 如果GMEM对应空间中仍有待拷贝的数据

    int stage = (As_Bs_K_index / BK) % numStages; // k+1一定为BK的倍数

    store_tile_A<BM, BK, BLOCK_SIZE>(A, As[stage], M, K, r0, As_Bs_K_index, tid, pipe);
    store_tile_B<BN, BK, BLOCK_SIZE>(B, Bs[stage], K, N, c0, As_Bs_K_index, tid, pipe);
    }

    // 包含未prefetch的空组。保证计数对齐
    pipe.producer_commit();


    pipe.consumer_wait();
    __syncthreads();

    const int col0 = c_thread_x * (sizeof(float4) / sizeof(float)); // 0~7  *  8
    const int stage = (k / BK) % numStages;

    #pragma unroll
    for (int p = 0; p < BK; p += 4) {

    // A: 每个输出 row 一条 LDS.128，一次拿到 p=pg..pg+3
    #pragma unroll
    for (int i = 0; i < Tm; ++i) {

    int row = c_thread_y + i * C_BLOCK_Y;

    FLOAT4(*(a_frag[i])) = FLOAT4(As[stage][row][
    swizzle_col<BK / 4>(row, p)
    ]);
    }

    // B: 每个 p 一条 LDS.128
    #pragma unroll
    for (int pp = 0; pp < 4; ++pp) {

    #pragma unroll
    for (int offset = 0; offset < Tn; offset += 4) { // 每个线程的b_frag数组共Tn个位置，存放 Tn / 4 个float4数据

    FLOAT4(*(b_frag + offset)) = FLOAT4(Bs[stage][p + pp][col0 + offset * C_BLOCK_X]);
    }

    // 每读取完一行Bs，立即计算
    #pragma unroll
    for (int i = 0; i < Tm; ++i) {

    #pragma unroll
    for (int j = 0; j < Tn; ++j) {

    Ct[i][j] += a_frag[i][pp] * b_frag[j];
    }
    }
    }
    }



    __syncthreads(); //避免在本轮循环计算完成前，SMEM被下一轮数据覆盖
    pipe.consumer_release();
    }


    // 写回结果
    #pragma unroll
    for (int i = 0; i < Tm; i++) {

    int r = r0 + c_thread_y + i * C_BLOCK_Y;
    int c = c0 + c_thread_x * ( sizeof(float4) / sizeof(float) );

    if (N % 4 == 0) {

        if (r < M) {

            #pragma unroll
            for (int offset = 0; offset < Tn; offset += 4) {

                const int col = c + offset * C_BLOCK_X;
                if (col + 3 < N) {
                    FLOAT4(C[r * N + col]) = FLOAT4(Ct[i][offset]);
                }
            }
        }
    }
    else {

        if (r < M) {

            #pragma unroll
            for (int offset = 0; offset < Tn; ++offset) {

                const int col = c + (offset / 4) * (C_BLOCK_X * 4) + offset % 4;
                if (col < N) {
                    C[r * N + col] = Ct[i][offset];
                }
            }
        }
    }
    }
}





} // namespace CUDAkernels
