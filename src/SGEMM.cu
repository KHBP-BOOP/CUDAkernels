
#include <cuda_runtime_api.h>

#define FLOAT4(f) *reinterpret_cast<float4*>(&f)


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

        //每个thread在block中的索引
        int r = r0 + i, c = k + a_thread_x * 4;

        //计算线程块尺寸时应采用向上取整的除法，故存在线程数量大于数据总量的情况
        float4 temp = (r < M && c + 3 < K) ? FLOAT4(A[r * K + c]) : make_float4(0.f, 0.f, 0.f, 0.f);

        // 同一block内，具体一次K维度的循环中，所有thread的r0一定、k一定
        FLOAT4(As[i][a_thread_x * 4]) = temp; // STS.128
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

    // float4向量化前：8 * 32个线程，每个线程做4次LDG.32，8 * 128个数据循环4次处理完成
    // float4向量化后：8 * 32个线程，每个线程做1次LDG.128，8 * 128个数据1次即可处理完成
    #pragma unroll
    for (int j = b_thread_x; j < BN; j += B_BLOCK_X) {

        //每个thread在block中的索引
        int r = k + b_thread_y, c = c0 + j * 4;

        //计算线程块尺寸时应采用向上取整的除法，故存在线程数量大于数据总量的情况
        float4 temp = (r < K && c + 3 < N) ? FLOAT4(B[r * N + c]) : make_float4(0.f, 0.f, 0.f, 0.f);

        // 同一block内，具体一次K维度的循环中，所有thread的c0一定、k一定
        FLOAT4(Bs[b_thread_y][j * 4]) = temp; // STS.128
    }
}


template <int BM = 128, int BN = 128, int BK = 8,
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


            // 寄存器上做外积
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    c_frag[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }

        __syncthreads();
    }


    //写回全局内存 - 方案一

    // 寄存器写回SMEM
    // 用 warp/lane 坐标，与compute部分一致

    // 一个线程负责一个数据间相邻的8*8部分
    constexpr int Cs_Y = WARP_Y * TM;
    __shared__ float Cs[Cs_Y][BN]; // 32 * 128    

    // 延续从SMEM加载到寄存器过程的warp级分块方案，由两个4 * 8warp tile以行优先方式排列组成尺寸为4 * 16的tile
    // 上文中tile的行、列索引：
    int thread_row_in_warp_tile = lane_row; // 0 ~ 3
    int thread_col_in_warp_tile = warp_col * WARP_X + lane_col; // 0 ~ 15

    constexpr int CHUNK_NUM = BM / Cs_Y; // BM、BN为自取值，无需向上取整除法
    for (int chunk = 0; chunk < CHUNK_NUM; ++chunk) {

        if (warp_row == chunk) {

            // 寄存器写回SMEM过程
            for (int m = 0; m < TM; m++) {
                    
                int r = thread_row_in_warp_tile * TM + m;
                for (int n = 0; n < TN; n++) {
                    
                    int c = thread_col_in_warp_tile * TN + n;
                    Cs[r][c] = (r < Cs_Y && c < BN) ? c_frag[m][n] : 0.0f;
                }
            }
            
            __syncthreads();

            
            // SMEM写回全局内存过程

            // 全局内存读取SMEM时线程重拍，线程块尺寸：
            constexpr int GLDC_C_BLOCK_X = BN / 4; // = 32
            constexpr int GLDC_C_BLOCK_Y = (2 * WARP_Y * WARP_X) / GLDC_C_BLOCK_X; // = 2

            // 线程索引重拍，tile尺寸变为2 * 32
            int GLDC_C_thread_y = lane_row / (WARP_Y /GLDC_C_BLOCK_Y); // 0 ~ 1
            int GLDC_C_thread_x = warp_col * WARP_X + lane_col
                + (GLDC_C_thread_y % 2) * (2 * WARP_X); // 0 ~ 31

            for (int i = 0; i < Cs_Y; i += GLDC_C_BLOCK_Y) {

                int r = i * GLDC_C_BLOCK_Y + GLDC_C_thread_y;
                int c = GLDC_C_thread_x * 4;

                float4 temp = FLOAT4(Cs[GLDC_C_thread_y][GLDC_C_thread_x * 4]);
                FLOAT4(C[r * N + c]) = temp;
            }

        }

        __syncthreads();
        
    }



    //写回全局内存 - 方案二
    // 写回用 warp/lane 坐标，与 compute 一致
    // 一个线程负责一个数据间相邻的8*8部分
    // 8*16
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
