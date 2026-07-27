
#include <cuda_runtime_api.h>


// 协作加载 tileA: 全局内存 → Shared Memory
// r0 = blockIdx.y * BM,  k = 当前 K 维度起点
template <int BM, int BK, int BLOCK_SIZE>
__device__ void load_tile_A(const float *__restrict__ A, float As[BM][BK],
                            int M, int K, int r0, int k, int tid)
{
    // 线程重排: 32×8 布局, 实现合并访问 (coalesced access)
    constexpr int A_BLOCK_X = BK;                     // = 8
    constexpr int A_BLOCK_Y = BLOCK_SIZE / A_BLOCK_X; // = 32
    int a_thread_x = tid % A_BLOCK_X;                 // 0 ~ 7
    int a_thread_y = tid / A_BLOCK_X;                 // 0 ~ 31

    #pragma unroll
    for (int i = a_thread_y; i < BM; i += A_BLOCK_Y)
    {
        int r = r0 + i, c = k + a_thread_x;
        As[i][a_thread_x] = (r < M && c < K) ? A[r * K + c] : 0.0f;
    }
}


// c0 = blockIdx.x * BN,  k = 当前 K 维度起点
template <int BN, int BK, int BLOCK_SIZE>
__device__ void load_tile_B(const float *__restrict__ B, float Bs[BK][BN],
                            int K, int N, int c0, int k, int tid)
{
    // 线程重排: 8×32 布局, 实现合并访问
    constexpr int B_BLOCK_X = 32;
    constexpr int B_BLOCK_Y = BLOCK_SIZE / B_BLOCK_X; // = 8
    int b_thread_x = tid % B_BLOCK_X;                 // 0 ~ 31
    int b_thread_y = tid / B_BLOCK_X;                 // 0 ~ 7

    #pragma unroll
    for (int j = b_thread_x; j < BN; j += B_BLOCK_X) {

        int r = k + b_thread_y, c = c0 + j;
        // 同一block内，具体一次K维度的循环中，所有thread的c0一定、k一定
        Bs[b_thread_y][j] = (r < K && c < N) ? B[r * N + c] : 0.0f;
    }
}


template <int BM, int BN, int BK,
    int BLOCK_SIZE, int Wx, int Wy,
    int TM, int TN>
__global__ void sgemm_thread_tiling(const float *A, const float *B, float *C, int M, int N, int K)
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

        // 协作加载 A、B 到 Shared Memory
        load_tile_A<BM, BK, BLOCK_SIZE>(A, As, M, K, by * BM, bk, tid);
        load_tile_B<BN, BK, BLOCK_SIZE>(B, Bs, K, N, bx * BN, bk, tid);

        __syncthreads();

        // 外积累加
        for (int k = 0; k < BK; k++) {

            // 从 Shared Memory 加载到寄存器

            //1个线程读取1个数据至a_frag
            a_frag[lane_col] = As[warp_row * WARP_Y * TM + lane_row * TM + lane_col][k];

            #pragma unroll
            for (int i = 0; i < WARP_X; i++) {
                
                
                a_frag[i] = __shfl_sync(0xffffffff, a_frag[i], i, 8);
            }

            //1个线程读取2个数据至b_frag
            b_frag[lane_row * 2] = Bs[k][warp_col * WARP_X * TN + lane_col * TN + lane_row * 2];
            b_frag[lane_row * 2 + 1] = Bs[k][warp_col * WARP_X * TN + lane_col * TN + lane_row * 2 + 1];
 
            
            // 广播 b_frag
            #pragma unroll
            for (int j = 0; j < TN; j += 2) {
                
                b_frag[j] = __shfl_sync(0xffffffff, b_frag[j], j * WARP_X);
                b_frag[j + 1] = __shfl_sync(0xffffffff, b_frag[j + 1], j * WARP_X);

            }

            #pragma unroll
            for (int j = 0; j < WARP_Y; j++) {
                
                b_frag[j * 2]     = __shfl_sync(0xffffffff, b_frag[j * 2],     j * WARP_X + lane_col);
                b_frag[j * 2 + 1] = __shfl_sync(0xffffffff, b_frag[j * 2 + 1], j * WARP_X + lane_col);
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
