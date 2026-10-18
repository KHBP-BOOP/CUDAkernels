
#include <cuda_runtime_api.h>

#include <iostream>


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

// v5
// 协作加载 tileA: 全局内存 → Shared Memory
// r0 = blockIdx.y * BM,  k = 当前 K 维度起点
template <int BM, int BK, int BLOCK_SIZE>
__device__ void load_tile_A_to_transposed_As(const float *__restrict__ A, float As[BK][BM],
                            int M, int K, int r0, int k, int tid)
{
    // 线程重排: 128 * 2 布局, 实现合并访问以及float4向量化 (coalesced access)
    constexpr int A_BLOCK_X = BK / 4;                     // = 2
    constexpr int A_BLOCK_Y = BLOCK_SIZE / A_BLOCK_X; // = 128
    int a_thread_x = tid % A_BLOCK_X;                 // 0 ~ 1
    int a_thread_y = tid / A_BLOCK_X;                 // 0 ~ 127

    // float4向量化前：32 * 8个线程，每个线程做4次LDG.32，128 * 8个数据循环4次处理完成
    // float4向量化后：128 * 2个线程，每个线程做1次LDG.128，128 * 8个数据1次即可处理完成
    #pragma unroll
    for (int i = a_thread_y; i < BM; i += A_BLOCK_Y) {

        // 每个thread在block中的索引
        int r = r0 + i, c = k + a_thread_x * 4;

        // 计算线程块尺寸时应采用向上取整的除法，故存在线程数量大于数据总量的情况
        // 为确保LDG.128的16B对齐，K必须是4的倍数（最优情况下是 BK=8 的倍数）
        // k为8的倍数，故c一定为4的倍数；c、K均为4的倍数，故不存在数据丢失情况
        float4 temp = (r < M && c + 3 < K) ? CONST_FLOAT4(A[r * K + c]) : make_float4(0.f, 0.f, 0.f, 0.f); // LDG.128

        // 转置存入As，将STS.128拆分为4次标量STS.32
        As[a_thread_x * 4 + 0][i] = temp.x;
        As[a_thread_x * 4 + 1][i] = temp.y;
        As[a_thread_x * 4 + 2][i] = temp.z;
        As[a_thread_x * 4 + 3][i] = temp.w;

        //此时STS一步出现2way冲突，但As存储至afrag时，原先需循环BK次、每次均为8way的冲突消失
    }
}






// v2
// 协作加载 tileA: 全局内存 → Shared Memory
// r0 = blockIdx.y * BM,  k = 当前 K 维度起点
template <int BM, int BK, int BLOCK_SIZE>
__device__ void load_tile_A(const float *__restrict__ A, float As[BM][BK],
                            int M, int K, int r0, int k, int tid)
{
    // 线程重排: 128 * 2 布局, 实现合并访问以及float4向量化 (coalesced access)
    constexpr int A_BLOCK_X = BK / 4;                     // = 8
    constexpr int A_BLOCK_Y = BLOCK_SIZE / A_BLOCK_X; // = 32
    constexpr int A_TILES = (BM + A_BLOCK_Y - 1) / A_BLOCK_Y;

    int a_thread_x = tid % A_BLOCK_X;                 // 0 ~ 7
    int a_thread_y = tid / A_BLOCK_X;                 // 0 ~ 31


    // float4向量化前：32 * 8个线程，每个线程做4次LDG.32，128 * 8个数据循环4次处理完成
    // float4向量化后：128 * 2个线程，每个线程做1次LDG.128，128 * 8个数据1次即可处理完成


    // 计算线程块尺寸时应采用向上取整的除法，故存在线程数量大于数据总量的情况
    // 为确保LDG.128的16B对齐，K必须是4的倍数（最优情况下是 BK=8 的倍数）
    if (K % 4 == 0) {

        int a_thread_x_float4 = a_thread_x * 4;
        int c = k + a_thread_x_float4;

        #pragma unroll
        for (int t = 0; t < A_TILES; ++t) { //BM 128
            int i = a_thread_y + t * A_BLOCK_Y;
            if (i < BM) {

                int r = r0 + i;

                //同一block内，具体一次K维度的循环中，所有thread的r0一定、k一定。
                float4 temp = (r < M && c + 3 < K) ? CONST_FLOAT4(A[r * K + c]) : make_float4(0.f, 0.f, 0.f, 0.f); // LDG.128

                // 同一block内，具体一次K维度的循环中，所有thread的r0一定、k一定
                FLOAT4(As[i][
                    swizzle_col<BK / 4>(i, a_thread_x_float4)
                ]) = temp; // STS.128
                // (r/16) | (c%8) ^ r %  (r/16)

                //warp32个线程中，每个线程处理一个float4类型，共512B；
                //硬件将其划分为4个128B内存事务，每个事务均占满SMEM的32个bank，流水线串行执行事务时无bank conflict
            }
        }
    }
    else {


        #pragma unroll
        for (int i = a_thread_y; i < BM; i += A_BLOCK_Y) { //BM 128

            int r = r0 + i, c = k + a_thread_x; // 128 128
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
                            int K, int N, int c0, int k, int tid)
{
    // Bs   32 * 64
    // 线程重排: 8×32 布局, 实现合并访问
    constexpr int B_BLOCK_X = BN / 4; // 16
    constexpr int B_BLOCK_Y = BLOCK_SIZE / B_BLOCK_X; // = 16
    constexpr int B_TILES = (BK + B_BLOCK_Y - 1) / B_BLOCK_Y;
    int b_thread_x = tid % B_BLOCK_X;                 // 0 ~ 15
    int b_thread_y = tid / B_BLOCK_X;                 // 0 ~ 15

    // float4向量化前：8 * 32个线程，每个线程做4次LDG.32，8 * 128个数据循环4次处理完成
    // float4向量化后：8 * 32个线程，每个线程做1次LDG.128，8 * 128个数据1次即可处理完成


    if (N % 4 == 0) {

        int b_thread_x_float4 = b_thread_x * 4;
        int c = c0 + b_thread_x_float4;


        #pragma unroll
        for (int t = 0; t < B_TILES; ++t) {

            int j = b_thread_y + t * B_BLOCK_Y;
            if (j < BK) {

                int r = k + j;
                //同一block内，具体一次K维度的循环中，所有thread的c0一定、k一定

                // 计算线程块尺寸时应采用向上取整的除法，故存在线程数量大于数据总量的情况
                // 为确保LDG.128的16B对齐，N必须是4的倍数（最优情况下是 BN=128 的倍数）
                // c0 = bx * BN，故c一定为4的倍数，不存在数据丢失情况
                float4 temp = (r < K && c + 3 < N) ? CONST_FLOAT4(B[r * N + c]) : make_float4(0.f, 0.f, 0.f, 0.f); // LDG.128

                // 同一block内，具体一次K维度的循环中，所有thread的c0一定、k一定
                FLOAT4(Bs[j][b_thread_x_float4]) = temp; // STS.128
            }
        }



        
    }
    else {

        #pragma unroll
        for (int j = b_thread_x; j < BN; j += B_BLOCK_X) {

            int r = k + b_thread_y, c = c0 + j;
            //同一block内，具体一次K维度的循环中，所有thread的c0一定、k一定

            Bs[b_thread_y][j] = (r < K && c < N) ? B[r * N + c] : 0.0f;
        }

    }
}





// v2
template <int BM, int BN, int BK, int BLOCK_SIZE>
__global__ void sgemm_block_tiling(const float* A, const float* B, float* C,
int M, int N, int K) {
    // 16B 对齐：两个 fragment 都按 float4 读取（LDS.128）
    __shared__ __align__(16) float As[BM][BK]; // 64 32
    __shared__ __align__(16) float Bs[BK][BN]; // 32 64

    int r0 = blockIdx.y * BM;
    int c0 = blockIdx.x * BN;
    int tid = threadIdx.x;

    // 计算 tileC 、写入C 时的线程排布（16×16）
    constexpr int C_BLOCK_X = 16;
    constexpr int C_BLOCK_Y = BLOCK_SIZE / C_BLOCK_X;  // = 16
    int c_thread_x = tid % C_BLOCK_X; // 0 ~ 15
    int c_thread_y = tid / C_BLOCK_X; // 0 ~ 15

    // 16 * 16 threads 负责 64 * 64 个元素
    // 每个线程负责 Tm×Tn 个输出元素
    constexpr int Tm = BM / C_BLOCK_Y;  // = 4 跨步覆盖64行 由BM、BLOCK_SIZE决定
    constexpr int Tn = BN / C_BLOCK_X;  // = 4 跨步覆盖64列 由BN决定
    float Ct[Tm][Tn] = { 0.0f };

    float a_frag[Tm] = { 0.0f };
    float b_frag[Tn] = { 0.0f };


    // K-Loop
    //一次循环对应
    for (int k = 0; k < K; k += BK) {

        load_tile_A<BM, BK, BLOCK_SIZE>(A, As, M, K, r0, k, tid);
        load_tile_B<BN, BK, BLOCK_SIZE>(B, Bs, K, N, c0, k, tid);

        //确保SMEM数据为本轮循环的数据
        __syncthreads();


        // 外积方式计算 As × Bs
        // 1个thread 16 个元素
        #pragma unroll
        for (int p = 0; p < BK; p++) {


            // 从 Shared Memory 加载到寄存器
            #pragma unroll
            for (int i = 0; i < Tm; ++i) {

                int row = c_thread_y + i * C_BLOCK_Y;

                a_frag[i] = As[row][
                    swizzle_col<BK / 4>(row, p)
                ];
            }

            #pragma unroll
            for (int j = 0; j < Tn; ++j) {

                b_frag[j] = Bs[p][c_thread_x + j * C_BLOCK_X];
            }

            #pragma unroll
            for (int i = 0; i < Tm; ++i) {

                #pragma unroll
                for (int j = 0; j < Tn; ++j) {

                    Ct[i][j] += a_frag[i] * b_frag[j];
                    //每一个线程负责一对As中一个元素、Bs中一个元素的FMA运算
                }
            }
        }




        // // 外积方式计算 As × Bs
        // // 1个thread 16 个元素
        // #pragma unroll
        // /// 非类型模板参数BK为编译期常量
        // for (int p = 0; p < BK; p++) {
        //     for (int i = 0; i < Tm; i++) {
        //         int row = c_thread_y + i * C_BLOCK_Y; // 0 ~ 63
        //         //进入循环，16 * 16 个线程中，同一行线程计算出相同row，但不同于其他行
        //         for (int j = 0; j < Tn; j++) {
        //             int col = c_thread_x + j * C_BLOCK_X; // 0 ~ 63
        //             //同一列线程计算出相同col，但不同于其他列
        //             Ct[i][j] += As[row][p] * Bs[p][col];
        //             //4*4       64*4        8*128
        //             //每一个线程负责一对As中一个元素、Bs中一个元素的FMA运算
        //         }
        //     }
        // }

        __syncthreads(); //避免在本轮循环计算完成前，SMEM被下一轮数据覆盖
    }

    // 写回结果
    for (int i = 0; i < Tm; i++) {

        int r = r0 + c_thread_y + i * C_BLOCK_Y;
        for (int j = 0; j < Tn; j++) {
            
            int c = c0 + c_thread_x + j * C_BLOCK_X;
            if (r < M && c < N) {
                C[r * N + c] = Ct[i][j];
            }
        }
    }
}







} // namespace CUDAkernels