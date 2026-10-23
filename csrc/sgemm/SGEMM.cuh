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

// // 协作加载 tileA: 全局内存 → Shared Memory
// // r0 = blockIdx.y * BM,  k = 当前 K 维度起点
// template <int BM, int BK, int BLOCK_SIZE>
// __device__ void load_tile_A_to_transposed_As(const float *__restrict__ A, float As[BK][BM],
//                             int M, int K, int r0, int k, int tid)
// {
//     // 线程重排: 128 * 2 布局, 实现合并访问以及float4向量化 (coalesced access)
//     constexpr int A_BLOCK_X = BK / 4;                     // = 2
//     constexpr int A_BLOCK_Y = BLOCK_SIZE / A_BLOCK_X; // = 128
//     int a_thread_x = tid % A_BLOCK_X;                 // 0 ~ 1
//     int a_thread_y = tid / A_BLOCK_X;                 // 0 ~ 127
//
//     // float4向量化前：32 * 8个线程，每个线程做4次LDG.32，128 * 8个数据循环4次处理完成
//     // float4向量化后：128 * 2个线程，每个线程做1次LDG.128，128 * 8个数据1次即可处理完成
//     #pragma unroll
//     for (int i = a_thread_y; i < BM; i += A_BLOCK_Y) {
//
//         // 每个thread在block中的索引
//         int r = r0 + i, c = k + a_thread_x * 4;
//
//         // 计算线程块尺寸时应采用向上取整的除法，故存在线程数量大于数据总量的情况
//         // 为确保LDG.128的16B对齐，K必须是4的倍数（最优情况下是 BK=8 的倍数）
//         // k为8的倍数，故c一定为4的倍数；c、K均为4的倍数，故不存在数据丢失情况
//         float4 temp = (r < M && c + 3 < K) ? CONST_FLOAT4(A[r * K + c]) : make_float4(0.f, 0.f, 0.f, 0.f); // LDG.128
//
//         // 转置存入As，将STS.128拆分为4次标量STS.32
//         As[a_thread_x * 4 + 0][i] = temp.x;
//         As[a_thread_x * 4 + 1][i] = temp.y;
//         As[a_thread_x * 4 + 2][i] = temp.z;
//         As[a_thread_x * 4 + 3][i] = temp.w;
//
//         //此时STS一步出现2way冲突，但As存储至afrag时，原先需循环BK次、每次均为8way的冲突消失
//     }
// }



// v2
// 协作加载 tileA: 全局内存 → Shared Memory
// r0 = blockIdx.y * BM,  k = 当前 K 维度起点
template <int BM, int BK, int BLOCK_SIZE>
__device__ void load_tile_A(const float *__restrict__ A, float As[BM][BK],
                            int M, int K, int r0, int k0, int tid, cuda::pipeline<cuda::thread_scope_thread>& pipe)
{
    // 线程重排: 32 * 8 布局, 实现合并访问以及float4向量化 (coalesced access)
    constexpr int A_BLOCK_X = BK / 4; // = 8
    constexpr int A_BLOCK_Y = BLOCK_SIZE / A_BLOCK_X; // = 32
    constexpr int A_TILES = (BM + A_BLOCK_Y - 1) / A_BLOCK_Y;

    int a_thread_x = tid % A_BLOCK_X;                 // 0 ~ 7
    int a_thread_y = tid / A_BLOCK_X;                 // 0 ~ 31

    // 计算线程块尺寸时应采用向上取整的除法，故存在线程数量大于数据总量的情况
    if (K % 4 == 0) {

        int a_thread_x_float4 = a_thread_x * 4;
        int c = k0 + a_thread_x_float4;

        #pragma unroll
        for (int t = 0; t < A_TILES; ++t) { //BM 128
            int i = a_thread_y + t * A_BLOCK_Y;
            if (i < BM) {

                int r = r0 + i;

                // //同一block内，具体一次K维度的循环中，所有thread的r0一定、k一定
                // float4 temp = (r < M && c + 3 < K) ? CONST_FLOAT4(A[r * K + c]) : make_float4(0.f, 0.f, 0.f, 0.f); // LDG.128

                // // 同一block内，具体一次K维度的循环中，所有thread的r0一定、k一定
                // FLOAT4(As[i][
                //     swizzle_col<BK / 4>(i, a_thread_x_float4)
                // ]) = temp; // STS.128

                if (r < M && c + 3 < K) {

                    cuda::memcpy_async(
                        &FLOAT4(As[i][swizzle_col<BK / 4>(i, a_thread_x_float4)]),
                        &CONST_FLOAT4(A[r * K + c]),
                        cuda::aligned_size_t<16>(sizeof(float)),
                        pipe
                    );
                }
                else {

                    FLOAT4(As[i][
                        swizzle_col<BK / 4>(i, a_thread_x_float4)
                    ]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f); //STS.128
                }

            }
        }
    }
    else {


        #pragma unroll
        for (int i = a_thread_y; i < BM; i += A_BLOCK_Y) { //BM 128

            int r = r0 + i, c = k0 + a_thread_x; // 128 128
            //同一block内，具体一次K维度的循环中，所有thread的r0一定、k一定。
            As[i][a_thread_x] = (r < M && c < K) ? A[r * K + c] : 0.0f; //所有线程均运行该行代码，将HBM中的数据存入各自对应的block的SMEM
        }
    }
}

// v2
// 协作加载 tileB: 全局内存 → Shared Memory
// c0 = blockIdx.x * BN,  k = 当前 K 维度起点
template <int BN, int BK, int BLOCK_SIZE>
__device__ void load_tile_B(const float *__restrict__ B, float Bs[BK][BN],
                            int K, int N, int c0, int k0, int tid, cuda::pipeline<cuda::thread_scope_thread>& pipe)
{
    // Bs   32 * 64
    // 线程重排: 16 * 16 布局, 实现合并访问
    constexpr int B_BLOCK_X = BN / 4; // 16
    constexpr int B_BLOCK_Y = BLOCK_SIZE / B_BLOCK_X; // = 16
    constexpr int B_TILES = (BK + B_BLOCK_Y - 1) / B_BLOCK_Y; // 2
    int b_thread_x = tid % B_BLOCK_X;                 // 0 ~ 15
    int b_thread_y = tid / B_BLOCK_X;                 // 0 ~ 15

    if (N % 4 == 0) {

        int b_thread_x_float4 = b_thread_x * 4;
        int c = c0 + b_thread_x_float4;


        #pragma unroll
        for (int t = 0; t < B_TILES; ++t) {

            int j = b_thread_y + t * B_BLOCK_Y; // 0 ~ 31
            if (j < BK) {

                int r = k0 + j;
                //同一block内，具体一次K维度的循环中，所有thread的c0一定、k一定

                // // 计算线程块尺寸时应采用向上取整的除法，故存在线程数量大于数据总量的情况
                // float4 temp = (r < K && c + 3 < N) ? CONST_FLOAT4(B[r * N + c]) : make_float4(0.f, 0.f, 0.f, 0.f); // LDG.128

                // // 同一block内，具体一次K维度的循环中，所有thread的c0一定、k一定
                // FLOAT4(Bs[j][b_thread_x_float4]) = temp; // STS.128

                if (r < K && c + 3 < N) {

                    cuda::memcpy_async(
                        &FLOAT4(Bs[j][b_thread_x_float4]),
                        &CONST_FLOAT4(B[r * N + c]),
                        cuda::aligned_size_t<16>(sizeof(float)),
                        pipe
                    );
                }
                else {

                    FLOAT4(Bs[j][b_thread_x_float4]) = make_float4(0.f, 0.f, 0.f, 0.f);
                }
            }
        }



        
    }
    else {

        #pragma unroll
        for (int j = b_thread_x; j < BN; j += B_BLOCK_X) {

            int r = k0 + b_thread_y, c = c0 + j;
            //同一block内，具体一次K维度的循环中，所有thread的c0一定、k一定

            Bs[b_thread_y][j] = (r < K && c < N) ? B[r * N + c] : 0.0f;
        }

    }
}


// v2
template <int BM, int BN, int BK, int BLOCK_SIZE, int numStages>
__global__ void sgemm_block_tiling(const float* A, const float* B, float* C,
int M, int N, int K) {
    // 16B 对齐：两个 fragment 都按 float4 读取（LDS.128）
    __shared__ __align__(16) float As[numStages][BM][BK]; // 128 
    __shared__ __align__(16) float Bs[numStages][BK][BN]; //     64

    int r0 = blockIdx.y * BM;
    int c0 = blockIdx.x * BN;
    int tid = threadIdx.x;

    // 计算 tileC 、写入C 时的线程排布（8 * 16）
    constexpr int C_BLOCK_X = 8;
    constexpr int C_BLOCK_Y = BLOCK_SIZE / C_BLOCK_X;  // = 16
    int c_thread_x = tid % C_BLOCK_X; // 0 ~ 7
    int c_thread_y = tid / C_BLOCK_X; // 0 ~ 15

    // 16 * 8 threads 负责 128 * 64 个元素
    // 每个线程负责 Tm×Tn 个输出元素
    constexpr int Tm = BM / C_BLOCK_Y;  // = 8 跨步覆盖128行 由BM、BLOCK_SIZE决定
    constexpr int Tn = BN / C_BLOCK_X;  // = 8 跨步覆盖64列 由BN决定

    float Ct[Tm][Tn] = { 0.0f };
    float a_frag[Tm][4] = { 0.0f }; // 存放float4数据
    float b_frag[Tn] = { 0.0f }; // Tn==8，故优化为连续存储2个float4



    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();


    // 每个线程先异步拷贝*下一轮K循环中计算部分*需要用到的数据，然后执行*本轮K循环*的计算操作

    // pipeline为线程私有
    // 启动前，将前 numStagesBefore 个stage逐个占用，启动异步拷贝，后续以隐藏延迟
    int numStagesBefore = numStages - 1; // 1 待定，但一定小于等于numStages

    for (int stage = 0; stage < numStagesBefore; ++stage) {

        int k0 = stage * BK; // K方向上的起始地址
        load_tile_A<BM, BK, BLOCK_SIZE>(A, As[stage], M, K, r0, k0, tid, pipe);
        load_tile_B<BN, BK, BLOCK_SIZE>(B, Bs[stage], K, N, c0, k0, tid, pipe);

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

            load_tile_A<BM, BK, BLOCK_SIZE>(A, As[stage], M, K, r0, As_Bs_K_index, tid, pipe);
            load_tile_B<BN, BK, BLOCK_SIZE>(B, Bs[stage], K, N, c0, As_Bs_K_index, tid, pipe);
        }

        // 包含未prefetch的空组。保证计数对齐
        pipe.producer_commit();


        pipe.consumer_wait();
        __syncthreads();

        const int col0 = c_thread_x * Tn; // 0~7  *  8
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

                    FLOAT4(*(b_frag + offset)) = FLOAT4(Bs[stage][p + pp][col0 + offset]);
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
        int c = c0 + c_thread_x * Tn;

        if (r < M && c < N) {

            #pragma unroll
            for (int offset = 0; offset < Tn; offset += 4) { // 每个线程的b_frag数组共Tn个位置，读取 Tn / 4 个float4数据

                FLOAT4(C[r * N + c + offset]) = FLOAT4(Ct[i][offset]);
            }
        }
    }
}


} // namespace CUDAkernels