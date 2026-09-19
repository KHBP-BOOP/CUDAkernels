#include <cuda_runtime_api.h>

#include <iostream>

#define FLOAT4(f) *reinterpret_cast<float4*>(&f)
#define CONST_FLOAT4(f) *reinterpret_cast<const float4*>(&f)


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

// v3 v4
// 协作加载 tileA: 全局内存 → Shared Memory
// r0 = blockIdx.y * BM,  k = 当前 K 维度起点
template <int BM, int BK, int BLOCK_SIZE>
__device__ void load_tile_A(const float *__restrict__ A, float As[BM][BK],
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

        // 同一block内，具体一次K维度的循环中，所有thread的r0一定、k一定
        FLOAT4(As[i][a_thread_x * 4]) = temp; // STS.128

        //warp32个线程中，每个线程处理一个float4类型，共512B；
        //硬件将其划分为4个128B内存事务，每个事务均占满SMEM的32个bank，流水线串行执行事务时无bank conflict
    }
}


// 协作加载 tileB: 全局内存 → Shared Memory
// c0 = blockIdx.x * BN,  k = 当前 K 维度起点
template <int BN, int BK, int BLOCK_SIZE>
__device__ void load_tile_B(const float *__restrict__ B, float Bs[BK][BN],
                            int K, int N, int c0, int k, int tid)
{
    // 线程重排: 8×32 布局, 实现合并访问
    constexpr int B_BLOCK_X = 128 / 4;
    constexpr int B_BLOCK_Y = BLOCK_SIZE / B_BLOCK_X; // = 8
    int b_thread_x = tid % B_BLOCK_X;                 // 0 ~ 31
    int b_thread_y = tid / B_BLOCK_X;                 // 0 ~ 7

    // float4向量化前：8 * 32个线程，每个线程做4次LDG.32，8 * 128个数据循环4次处理完成
    // float4向量化后：8 * 32个线程，每个线程做1次LDG.128，8 * 128个数据1次即可处理完成
    #pragma unroll
    for (int j = b_thread_y; j < BK; j += B_BLOCK_Y) {

        // 每个thread在block中的索引
        int r = k + j, c = c0 + b_thread_x * 4;

        // 计算线程块尺寸时应采用向上取整的除法，故存在线程数量大于数据总量的情况
        // 为确保LDG.128的16B对齐，N必须是4的倍数（最优情况下是 BN=128 的倍数）
        // c0 = bx * BN，故c一定为4的倍数，不存在数据丢失情况
        float4 temp = (r < K && c + 3 < N) ? CONST_FLOAT4(B[r * N + c]) : make_float4(0.f, 0.f, 0.f, 0.f); // LDG.128

        // 同一block内，具体一次K维度的循环中，所有thread的c0一定、k一定
        FLOAT4(Bs[j][b_thread_x * 4]) = temp; // STS.128
    }
}








// v1
template <int BM, int BN, int BK, int BLOCK_SIZE>
__global__ void sgemm_block_tiling_v1(const float* A, const float* B, float* C,
                                   int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int r0 = blockIdx.y * BM;//
    int c0 = blockIdx.x * BN;//
    int tid = threadIdx.x;

    // 加载 tileA 时的线程重排
    constexpr int A_BLOCK_X = BK;  // = 4
    constexpr int A_BLOCK_Y = BLOCK_SIZE / A_BLOCK_X;  // = 64
    int a_thread_x = tid % A_BLOCK_X; // 0 ~ 3
    int a_thread_y = tid / A_BLOCK_X; // 0 ~ 63

    // 加载 tileB 时的线程重排
    constexpr int B_BLOCK_X = BN;  // = 64
    constexpr int B_BLOCK_Y = BLOCK_SIZE / B_BLOCK_X;  // = 4
    int b_thread_x = tid % B_BLOCK_X;
    int b_thread_y = tid / B_BLOCK_X;

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

    // K-Loop
    //一次循环对应
    for (int k = 0; k < K; k += BK) {

        //r0用于A、C矩阵行索引，c0用于B、C矩阵列索引
        //A矩阵列索引、B矩阵行索引借助K-Loop中的循环变量k
        int r = r0 + a_thread_y;
        int c = k + a_thread_x;


        // 将tileA数据载入SMEM
        //同一block内，具体一次K维度的循环中，所有thread的r0一定、k一定。
        As[a_thread_y][a_thread_x] = (r < M && c < K) ? A[r * K + c] : 0.0f; //所有线程均运行该行代码，将HBM中的数据存入各自对应的block的SMEM

        // 将tileB数据载入SMEM
        r = k + b_thread_y;
        c = c0 + b_thread_x;
        //同一block内，具体一次K维度的循环中，所有thread的c0一定、k一定
        Bs[b_thread_y][b_thread_x] = (r < K && c < N) ? B[r * N + c] : 0.0f;
        


        //确保SMEM数据为本轮循环的数据
        __syncthreads();



        // 外积方式计算 As × Bs
        // 1个thread 16 个元素
        #pragma unroll
        for (int p = 0; p < BK; p++) {
            for (int i = 0; i < Tm; i++) {
                int row = c_thread_y + i * C_BLOCK_Y; //0~120
                //进入循环，16 * 16 个线程中，同一行线程计算出相同row，但不同于其他行
                for (int j = 0; j < Tn; j++) {
                    int col = c_thread_x + j * C_BLOCK_X; //0~120 
                    //同一列线程计算出相同col，但不同于其他列
                    Ct[i][j] += As[row][p] * Bs[p][col];
                    //4*4       64*4        8*128
                    //每一个线程负责一对As中一个元素、Bs中一个元素的FMA运算
                }
            }
        }
    
        __syncthreads(); //避免在本轮循环计算完成前，SMEM被下一轮数据覆盖
    }

    // 写回结果
    for (int i = 0; i < Tm; i++) {
        int r = r0 + c_thread_y + i * C_BLOCK_Y;
        for (int j = 0; j < Tn; j++) {
            int c = c0 + c_thread_x + j * C_BLOCK_X;
            if (r < M && c < N) C[r * N + c] = Ct[i][j];
        }
    }
}






// v3
template <int BM, int BN, int BK,
    int BLOCK_SIZE, int Wx, int Wy,
    int TM, int TN>
__global__ void sgemm_thread_tiling_v3(const float *A, const float *B, float *C, int M, int N, int K) {

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int tid = threadIdx.x;
    int by = blockIdx.y, bx = blockIdx.x;

    // 寄存器
    float a_frag[TM];
    float b_frag[TN];
    float c_frag[TM][TN] = {0.0f};


    int warp_id = tid >> 5; // tid / 32
    int lane_id = tid & 31; // tid % 32

    const int WARP_Y = Wy; // 4
    const int WARP_X = Wx; // 8
    const int C_BLOCK_X = BN / TN;
    const int C_BLOCK_Y = BM / TM;

    // 每一列 C_BLOCK_Y / WARP_Y 个 warp tile
    const int warp_tile_per_col = C_BLOCK_Y / WARP_Y;
    // 每一行 C_BLOCK_X / WARP_X 个 warp tile
    const int warp_tile_per_row = C_BLOCK_X / WARP_X;

    // warp 在 Block 中的位置（4×2 排列）
    int warp_row = warp_id / warp_tile_per_row; // / 2   M 方向：0~3
    int warp_col = warp_id % warp_tile_per_row; // % 2   N 方向：0~1

    // lane 在 warp 内的位置（4×8 排列，行主序）
    int lane_row = lane_id / WARP_X; // M 方向：0~3
    int lane_col = lane_id % WARP_X; // N 方向：0~7

    for (int bk = 0; bk < K; bk += BK)
    {

        // 协作加载 A、B 到 Shared Memory
        load_tile_A<BM, BK, BLOCK_SIZE>(A, As, M, K, by * BM, bk, tid);
        load_tile_B<BN, BK, BLOCK_SIZE>(B, Bs, K, N, bx * BN, bk, tid);

        __syncthreads();

        // 外积累加
        for (int k = 0; k < BK; k++) {

            // 从 Shared Memory 加载到寄存器

            //1个线程读取1个数据至a_frag
            a_frag[lane_col] = As[warp_row * WARP_Y * TM + lane_row * TM + lane_col][k];

            //广播a_frag
            #pragma unroll
            for (int i = 0; i < WARP_X; i++) {

                a_frag[i] = __shfl_sync(0xffffffff, a_frag[i], i, 8);
            }


            //1个线程读取2个数据至b_frag
            b_frag[lane_row * 2] = Bs[k][warp_col * WARP_X * TN + lane_col * TN + lane_row * 2];
            b_frag[lane_row * 2 + 1] = Bs[k][warp_col * WARP_X * TN + lane_col * TN + lane_row * 2 + 1];
            
            
            // 广播 b_frag
            // 每个线程遍历各自的b_frag，从同列对应线程的或自身的b_frag中获取数据，使得同列线程的b_frag均存有正确数据且一致
            #pragma unroll
            for (int j = 0; j < TN; j += 2) {
                
                b_frag[j] = __shfl_sync(0xffffffff, b_frag[j], j * WARP_X / 2 + lane_col);
                b_frag[j + 1] = __shfl_sync(0xffffffff, b_frag[j + 1], j * WARP_X / 2 + lane_col);

            }

            // if (lane_col == 0) {

            //     for (int i = 0; i < TM; i++) {
            //         a_frag[i] = As[warp_row * WARP_Y * TM + lane_row * TM + i][k];
            //     }
            // }
            

            // if (lane_row == 0) {
            //     for (int j = 0; j < TN; j++) {
            //         b_frag[j] = Bs[k][warp_col * WARP_X * TN + lane_col * TN + j];
            //     }
            // }

            // // 广播 a_frag
            // #pragma unroll
            // for (int i = 0; i < TM; i++) {
            //     a_frag[i] = __shfl_sync(0xffffffff, a_frag[i], lane_row * WARP_X);
            //     //每一行线程的源线程的索引跨步等距
                
            //     //a_frag[i] = __shfl_sync(0xffffffff, a_frag[i], 0, 8);
            // }

            // // 广播 b_frag
            // #pragma unroll
            // for (int j = 0; j < TN; j++) {
            //     b_frag[j] = __shfl_sync(0xffffffff, b_frag[j], lane_col);
            //     //每一列线程的源线程的索引相邻
            // }

            // 寄存器上做外积
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    c_frag[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }

        __syncthreads();
    }

    
    // 写回用 warp/lane 坐标，与 compute 一致
    // 一个线程负责一个数据间相邻的8*8部分
    int base_row = by * BM + warp_row * WARP_Y * TM + lane_row * TM;
    int base_col = bx * BN + warp_col * WARP_X * TN + lane_col * TN;
    for (int i = 0; i < TM; i++) {
        int r = base_row + i;
        for (int j = 0; j < TN; j++) {
            int c = base_col + j;
            if (r < M && c < N) C[r * N + c] = c_frag[i][j];
        }
    }

}


// v4
template <int BM = 128, int BN = 128, int BK = 8,
    int BLOCK_SIZE, int Wx, int Wy,
    int TM, int TN>
__global__ void sgemm_thread_tiling_v4(const float *A, const float *B, float *C, int M, int N, int K)
{
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int tid = threadIdx.x;
    int by = blockIdx.y, bx = blockIdx.x;

    // 寄存器
    float a_frag[TM];
    float b_frag[TN];
    float c_frag[TM][TN] = {0.0f};


    int warp_id = tid >> 5; // tid / 32
    int lane_id = tid & 31; // tid % 32

    constexpr int WARP_Y = Wy; // 4
    constexpr int WARP_X = Wx; // 8

    //寄存器读取SMEM时线程重排，线程块尺寸
    constexpr int RLDS_C_BLOCK_X = BN / TN;
    constexpr int RLDS_C_BLOCK_Y = BM / TM;

    // 每一列 LDS_C_BLOCK_Y / WARP_Y 个 warp tile
    constexpr int warp_tile_per_col = RLDS_C_BLOCK_Y / WARP_Y;
    // 每一行 LDS_C_BLOCK_X / WARP_X 个 warp tile
    constexpr int warp_tile_per_row = RLDS_C_BLOCK_X / WARP_X;

    // warp 在 Block 中的位置（4×2 排列）
    int warp_row = warp_id / warp_tile_per_row; // / 2   M 方向：0~3
    int warp_col = warp_id % warp_tile_per_row; // % 2   N 方向：0~1

    // lane 在 warp 内的位置（4×8 排列，行主序）
    int lane_row = lane_id / WARP_X; // M 方向：0~3
    int lane_col = lane_id % WARP_X; // N 方向：0~7

    //K-Loop
    for (int bk = 0; bk < K; bk += BK)
    {

        // 协作加载 A、B 到 Shared Memory
        load_tile_A<BM, BK, BLOCK_SIZE>(A, As, M, K, by * BM, bk, tid);
        load_tile_B<BN, BK, BLOCK_SIZE>(B, Bs, K, N, bx * BN, bk, tid);

        __syncthreads();

        // 外积累加
        for (int k = 0; k < BK; k++) {

            // 从 Shared Memory 加载到寄存器

            //1个线程读取1个数据至a_frag
            a_frag[lane_col] = As[warp_row * WARP_Y * TM + lane_row * TM + lane_col][k];

            // 广播 a_frag
            // 每个线程遍历各自的a_frag，从同行对应线程的或自身的a_frag中获取数据，使得同行线程的a_frag均存有正确数据且一致
            #pragma unroll
            for (int i = 0; i < TM; i++) {

                a_frag[i] = __shfl_sync(0xffffffff, a_frag[i], i, 8);
            }

            //1个线程读取2个数据至b_frag
            b_frag[lane_row * 2] = Bs[k][warp_col * WARP_X * TN + lane_col * TN + lane_row * 2];
            b_frag[lane_row * 2 + 1] = Bs[k][warp_col * WARP_X * TN + lane_col * TN + lane_row * 2 + 1];
 
            
            // 广播 b_frag
            // 每个线程遍历各自的b_frag，从同列对应线程的或自身的b_frag中获取数据，使得同列线程的b_frag均存有正确数据且一致
            #pragma unroll
            for (int j = 0; j < TN; j += 2) {
                
                b_frag[j] = __shfl_sync(0xffffffff, b_frag[j], j * WARP_X / 2 + lane_col);
                b_frag[j + 1] = __shfl_sync(0xffffffff, b_frag[j + 1], j * WARP_X / 2 + lane_col);

            }

            // 寄存器上做外积
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    c_frag[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }

        __syncthreads();
    }


    // 写回全局内存，使用 warp/lane 坐标，与 compute 一致
    // 一个线程负责一个数据间相邻的8*8部分
    int base_row = by * BM + warp_row * WARP_Y * TM + lane_row * TM;
    int base_col = bx * BN + warp_col * WARP_X * TN + lane_col * TN;
    for (int i = 0; i < TM; i++) {

        int r = base_row + i;
        if (r >= M) {
            //M不为8的倍数时，负责余下的不足8行数据的线程同样循环8次，但没有数据的几次循环会执行continue语句跳过
            continue;
        }

        if (base_col + TN <= N) {
            
            //2条STG.128指令
            
            float4 temp = make_float4(c_frag[i][0], c_frag[i][1], c_frag[i][2], c_frag[i][3]);
            FLOAT4(C[r * N + base_col]) = temp;

            temp = make_float4(c_frag[i][4], c_frag[i][5], c_frag[i][6], c_frag[i][7]);
            FLOAT4(C[r * N + base_col + 4]) = temp;

        }

        else {

            // N不为8的倍数时，余下的不足8个的float数据逐个传输
            for (int j = 0; j < N - base_col; ++j) {
                C[r * N + base_col + j] = c_frag[i][j];
            }

        }

    }
}



// v5
template <int BM = 128, int BN = 128, int BK = 8,
    int BLOCK_SIZE, int Wx, int Wy,
    int TM, int TN>
__global__ void sgemm_thread_tiling_v5(const float *A, const float *B, float *C, int M, int N, int K)
{
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    int tid = threadIdx.x;
    int by = blockIdx.y, bx = blockIdx.x;

    // 寄存器
    float a_frag[TM];
    float b_frag[TN];
    float c_frag[TM][TN] = {0.0f};


    int warp_id = tid >> 5; // tid / 32
    int lane_id = tid & 31; // tid % 32

    constexpr int WARP_Y = Wy; // 4
    constexpr int WARP_X = Wx; // 8

    //寄存器读取SMEM时线程重排，线程块尺寸
    constexpr int RLDS_C_BLOCK_X = BN / TN;
    constexpr int RLDS_C_BLOCK_Y = BM / TM;

    // 每一列 LDS_C_BLOCK_Y / WARP_Y 个 warp tile
    constexpr int warp_tile_per_col = RLDS_C_BLOCK_Y / WARP_Y;
    // 每一行 LDS_C_BLOCK_X / WARP_X 个 warp tile
    constexpr int warp_tile_per_row = RLDS_C_BLOCK_X / WARP_X;

    // warp 在 Block 中的位置（4×2 排列）
    int warp_row = warp_id / warp_tile_per_row; // / 2   M 方向：0~3
    int warp_col = warp_id % warp_tile_per_row; // % 2   N 方向：0~1

    // lane 在 warp 内的位置（4×8 排列，行主序）
    int lane_row = lane_id / WARP_X; // M 方向：0~3
    int lane_col = lane_id % WARP_X; // N 方向：0~7

    //K-Loop
    for (int bk = 0; bk < K; bk += BK)
    {

        // 协作加载 A、B 到 Shared Memory
        load_tile_A_to_transposed_As<BM, BK, BLOCK_SIZE>(A, As, M, K, by * BM, bk, tid);
        load_tile_B<BN, BK, BLOCK_SIZE>(B, Bs, K, N, bx * BN, bk, tid);

        __syncthreads();

        // 外积累加
        for (int k = 0; k < BK; k++) {

            // 从 Shared Memory 加载到寄存器

            //1个线程读取1个float至a_frag
            //同一warp的32个线程发起1个内存事务，且数据严格连续对齐、无bank conflict
            a_frag[lane_col] = As[k][warp_row * WARP_Y * TM + lane_row * TM + lane_col];

            // 广播 a_frag
            // 每个线程遍历各自的a_frag，从同行对应线程的或自身的a_frag中获取数据，使得同行线程的a_frag均存有正确数据且一致
            #pragma unroll
            for (int i = 0; i < TM; i++) {

                a_frag[i] = __shfl_sync(0xffffffff, a_frag[i], i, 8);
            }


            // b_frag写入过程
            // 16个lane各读一个float4：相邻lane连续，无bank conflict
            float4 b_vec = make_float4(0.f, 0.f, 0.f, 0.f);
            if (lane_row < 2) {
                
                b_vec = CONST_FLOAT4(Bs[k][warp_col * WARP_X * TN + lane_id * 4]);
            }   

            // 全warp参与、无分歧：0~3来自lane 2*lane_col，4~7来自lane 2*lane_col+1
            // __shfl_sync()只支持32位或64位的基础标量类型，必须将float4类型数据拆开
            b_frag[0] = __shfl_sync(0xffffffff, b_vec.x, 2 * lane_col);
            b_frag[1] = __shfl_sync(0xffffffff, b_vec.y, 2 * lane_col);
            b_frag[2] = __shfl_sync(0xffffffff, b_vec.z, 2 * lane_col);
            b_frag[3] = __shfl_sync(0xffffffff, b_vec.w, 2 * lane_col);
            b_frag[4] = __shfl_sync(0xffffffff, b_vec.x, 2 * lane_col + 1);
            b_frag[5] = __shfl_sync(0xffffffff, b_vec.y, 2 * lane_col + 1);
            b_frag[6] = __shfl_sync(0xffffffff, b_vec.z, 2 * lane_col + 1);
            b_frag[7] = __shfl_sync(0xffffffff, b_vec.w, 2 * lane_col + 1);
            

            // 寄存器上做外积
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    c_frag[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }

        __syncthreads();
    }


    // 写回全局内存，使用 warp/lane 坐标，与 compute 一致
    // 一个线程负责一个数据间相邻的8*8部分
    int base_row = by * BM + warp_row * WARP_Y * TM + lane_row * TM;
    int base_col = bx * BN + warp_col * WARP_X * TN + lane_col * TN;
    for (int i = 0; i < TM; i++) {

        int r = base_row + i;
        if (r >= M) {
            //M不为8的倍数时，负责余下的不足8行数据的线程同样循环8次，但没有数据的几次循环会执行continue语句跳过
            continue;
        }

        if (base_col + TN <= N) {
            
            //2条STG.128指令
            
            float4 temp = make_float4(c_frag[i][0], c_frag[i][1], c_frag[i][2], c_frag[i][3]);
            FLOAT4(C[r * N + base_col]) = temp;

            temp = make_float4(c_frag[i][4], c_frag[i][5], c_frag[i][6], c_frag[i][7]);
            FLOAT4(C[r * N + base_col + 4]) = temp;

        }

        else {

            // N不为8的倍数时，余下的不足8个的float数据逐个传输
            for (int j = 0; j < N - base_col; ++j) {
                C[r * N + base_col + j] = c_frag[i][j];
            }

        }

    }
}





// 主机端启动封装：核函数模板在本翻译单元内完成实例化。
// 直接跨翻译单元链接 __global__ 模板实例化存在可见性问题
// （rdc=false 模式下模板实例化的 host stub 默认具有内部链接属性），
// 因此测试代码 (src/testSGEMM.cu) 通过本函数间接启动核函数

void launch_sgemm_thread_tiling_v1(const float *A, const float *B, float *C, int M, int N, int K) {

    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 4;
    constexpr int BLOCK_SIZE = 256;

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    
    sgemm_block_tiling_v1<BM, BN, BK, BLOCK_SIZE> <<<grid, block>>>(A, B, C, M, N, K);
}




void launch_sgemm_thread_tiling_v3(const float *A, const float *B, float *C,
                                int M, int N, int K)
{

    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 4;
    constexpr int BLOCK_SIZE = 256;

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_thread_tiling_v3<128, 128, 8, 256, 8, 4, 4, 4> <<<grid, block>>>(A, B, C, M, N, K);
    std::cout << "started sgemm_thread_tiling_v3 once." << std::endl;
}


void launch_sgemm_thread_tiling_v4(const float *A, const float *B, float *C,
                                int M, int N, int K)
{

    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 4;
    constexpr int BLOCK_SIZE = 256;

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);


    sgemm_thread_tiling_v4<128, 128, 8, 256, 8, 4, 8, 8> <<<grid, block>>>(A, B, C, M, N, K);
    std::cout << "started sgemm_thread_tiling_v4 once." << std::endl;

}


void launch_sgemm_thread_tiling_v5(const float *A, const float *B, float *C,
                                int M, int N, int K)
{

    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 4;
    constexpr int BLOCK_SIZE = 256;

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);


    sgemm_thread_tiling_v5<128, 128, 8, 256, 8, 4, 8, 8> <<<grid, block>>>(A, B, C, M, N, K);
    std::cout << "started sgemm_thread_tiling_v5 once." << std::endl;
}
