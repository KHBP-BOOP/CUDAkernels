
#include <cuda_runtime_api.h>

template <int BM, int BN, int BK, int BLOCK_SIZE>
__device__ void load_tile_A(float *A, float As[BM][BK], int M, int K, int r0, int k, int tid)
{
    // 加载 tileA 时的线程重排
    constexpr int A_BLOCK_X = BK;                     // 8
    constexpr int A_BLOCK_Y = BLOCK_SIZE / A_BLOCK_X; // = 32
    int a_thread_x = tid % A_BLOCK_X;                 // 0~7
    int a_thread_y = tid / A_BLOCK_X;                 // 0 ~ 31
// 将tileA数据载入SMEM（使用跨步循环覆盖 BM 行）
#pragma unroll
    for (int i = a_thread_y; i < BM; i += A_BLOCK_Y)
    {                                       // BM 128
        int r = r0 + i, c = k + a_thread_x; // 128 128
        // 同一block内，具体一次K维度的循环中，所有thread的r0一定、k一定。
        As[i][a_thread_x] = (r < M && c < K) ? A[r * K + c] : 0.0f; // 所有线程均运行该行代码，将HBM中的数据存入各自对应的block的SM
    }
}

template <int BM, int BN, int BK, int BLOCK_SIZE>
__device__ void load_tile_B(float *B, float Bs[BK][BN], int K, int N, int c0, int k, int tid)
{
    // 加载 tileB 时的线程重排
    constexpr int B_BLOCK_X = 32;
    constexpr int B_BLOCK_Y = BLOCK_SIZE / B_BLOCK_X; // = 8
    int b_thread_x = tid % B_BLOCK_X;
    int b_thread_y = tid / B_BLOCK_X;

// 协作加载 tileB（使用跨步循环覆盖 BN 列）
#pragma unroll
    for (int j = b_thread_x; j < BN; j += B_BLOCK_X)
    {
        int r = k + b_thread_y, c = c0 + j;
        // 同一block内，具体一次K维度的循环中，所有thread的c0一定、k一定
        Bs[b_thread_y][j] = (r < K && c < N) ? B[r * N + c] : 0.0f;
    }
}


template <int BM, int BN, int BK, int TM, int TN, int Wx, int Wy>
__global__ void sgemm_thread_tiling(const float *A, const float *B, float *C,
                                    int M, int N, int K)
{
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int tid = threadIdx.x;
    int by = blockIdx.y, bx = blockIdx.x;

    // 寄存器
    float a_frag[TM];
    float b_frag[TN];
    float c_frag[TM][TN] = {0.0f};

    // 每个线程在 C 中负责 TM×TN 的子块
    const int C_BLOCK_X = BN / TN;
    const int C_BLOCK_Y = BM / TM;
    int c_thread_x = tid % (BN / TN);
    int c_thread_y = tid / (BM / TM);
    int thread_row = c_thread_x * TM;
    int thread_col = c_thread_y * TN;

    int warp_id = tid >> 5; // tid / 32
    int lane_id = tid & 31; // tid % 32

    const int WARP_Y = Wy; // 4
    const int WARP_X = Wx; // 8

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

        // 协作加载 A、B 到 Shared Memory（省略边界检查）
        load_tile_A(A, As, by, bk, tid, M, K);
        load_tile_B(B, Bs, bx, bk, tid, K, N);

        __syncthreads();

        // 外积累加
        for (int k = 0; k < BK; k++) {

            // 从 Shared Memory 加载到寄存器

            if (lane_col == 0) {

                for (int i = 0; i < TM; i++) {
                    a_frag[i] = As[warp_row * WARP_Y * TM + lane_row * TM + i][k];
                }
            }

            if (lane_row == 0) {
                for (int j = 0; j < TN; j++) {
                    b_frag[j] = Bs[k][warp_col * WARP_X * TN + lane_col * TN + j];
                }
            }

            // 广播 a_frag
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                a_frag[i] = __shfl_sync(0xffffffff, a_frag[i], lane_row * WARP_X);
                //每一行线程的源线程的索引跨步等距
                
                //a_frag[i] = __shfl_sync(0xffffffff, a_frag[i], 0, 8);
            }

            // 广播 b_frag
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                b_frag[j] = __shfl_sync(0xffffffff, b_frag[j], lane_col);
                //每一列线程的源线程的索引相邻
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

    // 写回 Global Memory
    for (int i = 0; i < TM; i++)
    {
        int r = by * BM + i * C_BLOCK_Y + c_thread_y;
        for (int j = 0; j < TN; j++)
        {
            int c = bx * BN + j * C_BLOCK_X + c_thread_x;
            if (r < M && c < N)
                C[r * N + c] = c_frag[i][j];
        }
    }
}
