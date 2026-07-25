
#include <cuda_runtime_api.h>

template <int BM, int BN, int BK, int BLOCK_SIZE>
__global__ void sgemm_block_tiling_1(float* A, float* B, float* C,
                                   int M, int K, int N) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int r0 = blockIdx.y * BM;//
    int c0 = blockIdx.x * BN;//
    int tid = threadIdx.x;

    // 加载 tileA 时的线程重排
    constexpr int A_BLOCK_X = BK;  // = 8
    constexpr int A_BLOCK_Y = BLOCK_SIZE / A_BLOCK_X;  // = 32
    int a_thread_x = tid % A_BLOCK_X; // 0 ~ 7
    int a_thread_y = tid / A_BLOCK_X; // 0 ~ 31

    // 加载 tileB 时的线程重排
    constexpr int B_BLOCK_X = 32;
    constexpr int B_BLOCK_Y = BLOCK_SIZE / B_BLOCK_X;  // = 8
    int b_thread_x = tid % B_BLOCK_X;
    int b_thread_y = tid / B_BLOCK_X;

    // 计算 tileC 、写入C 时的线程排布（16×16）
    constexpr int C_BLOCK_X = 16;
    constexpr int C_BLOCK_Y = BLOCK_SIZE / C_BLOCK_X;  // = 16
    int c_thread_x = tid % C_BLOCK_X; // 0 ~ 15
    int c_thread_y = tid / C_BLOCK_X; // 0 ~ 15

    // 16 * 16 threads 负责 128 * 128 个元素
    // 每个线程负责 Tm×Tn 个输出元素
    constexpr int Tm = BM / C_BLOCK_Y;  // = 8 跨步覆盖128行 由BM、BLOCK_SIZE决定
    constexpr int Tn = BN / C_BLOCK_X;  // = 8 跨步覆盖128列 由BN决定
    float Ct[Tm][Tn] = {0.0f};

    // K-Loop
    //一次循环对应
    for (int k = 0; k < K; k += BK) {

        //r0用于A、C矩阵行索引，c0用于B、C矩阵列索引
        //A矩阵列索引、B矩阵行索引借助K-Loop中的循环变量k

        // 将tileA数据载入SMEM（使用跨步循环覆盖 BM 行）
        #pragma unroll
        for (int i = a_thread_y; i < BM; i += A_BLOCK_Y) { //BM 128
            // i            128 * 8    32 * 8
            // 0 32 64 96
            // 1 33 65 97
            // 2 34 66 98
            // ...
            // 31 63 95 127
            // tile A  8 * 32  256
            int r = r0 + i, c = k + a_thread_x; // 128 128
            //同一block内，具体一次K维度的循环中，所有thread的r0一定、k一定。
            As[i][a_thread_x] = (r < M && c < K) ? A[r * K + c] : 0.0f; //所有线程均运行该行代码，将HBM中的数据存入各自对应的block的SMEM
        }

        // 协作加载 tileB（使用跨步循环覆盖 BN 列）
        #pragma unroll
        for (int j = b_thread_x; j < BN; j += B_BLOCK_X) {
            int r = k + b_thread_y, c = c0 + j;
            //同一block内，具体一次K维度的循环中，所有thread的c0一定、k一定
            Bs[b_thread_y][j] = (r < K && c < N) ? B[r * N + c] : 0.0f;
        }

        //确保SMEM数据为本轮循环的数据
        __syncthreads();

        // 外积方式计算 As × Bs
        // 1个thread 64 个元素
        #pragma unroll
        for (int p = 0; p < BK; p++) {
            for (int i = 0; i < Tm; i++) {
                int row = c_thread_y + i * C_BLOCK_Y; //0~120
                //进入循环，16 * 16 个线程中，同一行线程计算出相同row，但不同于其他行
                for (int j = 0; j < Tn; j++) {
                    int col = c_thread_x + j * C_BLOCK_X; //0~120 
                    //同一列线程计算出相同col，但不同于其他列
                    Ct[i][j] += As[row][p] * Bs[p][col];
                    //8*8       128*8        8*128
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






template <int BM, int BN, int BK, int BLOCK_SIZE>
__device__ void load_tile_A(float* A, float As[BM][BK], int M, int K, int r0, int k, int tid) {
    // 加载 tileA 时的线程重排
    constexpr int A_BLOCK_X = BK;  //8
    constexpr int A_BLOCK_Y = BLOCK_SIZE / A_BLOCK_X;  // = 32
    int a_thread_x = tid % A_BLOCK_X; //0~7
    int a_thread_y = tid / A_BLOCK_X; // 0 ~ 31  
    // 将tileA数据载入SMEM（使用跨步循环覆盖 BM 行）    
    #pragma unroll
    for (int i = a_thread_y; i < BM; i += A_BLOCK_Y) { //BM 128                        
        int r = r0 + i, c = k + a_thread_x; // 128 128
        //同一block内，具体一次K维度的循环中，所有thread的r0一定、k一定。       
        As[i][a_thread_x] = (r < M && c < K) ? A[r * K + c] : 0.0f; //所有线程均运行该行代码，将HBM中的数据存入各自对应的block的SM
    }
}
template <int BM, int BN, int BK, int BLOCK_SIZE>
__device__ void load_tile_B(float* B, float Bs[BK][BN], int K, int N, int c0, int k, int tid) {               
    // 加载 tileB 时的线程重排
    constexpr int B_BLOCK_X = 32;
    constexpr int B_BLOCK_Y = BLOCK_SIZE / B_BLOCK_X;  // = 8
    int b_thread_x = tid % B_BLOCK_X;
    int b_thread_y = tid / B_BLOCK_X;

    // 协作加载 tileB（使用跨步循环覆盖 BN 列）
    #pragma unroll
    for (int j = b_thread_x; j < BN; j += B_BLOCK_X) {
        int r = k + b_thread_y, c = c0 + j;
        //同一block内，具体一次K维度的循环中，所有thread的c0一定、k一定
        Bs[b_thread_y][j] = (r < K && c < N) ? B[r * N + c] : 0.0f;
    }
}



template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_thread_tiling(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // 每个线程在 C 中负责 TM×TN 的子块
    const int thread_num = BM / TM * BN / TN;  // = 256
    int tid = threadIdx.x;
    int thread_row = (tid / (BN / TN)) * TM;
    int thread_col = (tid % (BN / TN)) * TN;

    // 寄存器存储
    float a_frag[TM];
    float b_frag[TN];
    float c_frag[TM][TN] = {0.0f};

    int by = blockIdx.y, bx = blockIdx.x;

    for (int bk = 0; bk < K; bk += BK) {
        // 协作加载 A、B 到 Shared Memory（省略边界检查）
        load_tile_A(A, As, by, bk, tid, M, K);
        load_tile_B(B, Bs, bx, bk, tid, K, N);
        __syncthreads();

        // 外积累加
        for (int k = 0; k < BK; k++) {
            // 从 Shared Memory 加载到寄存器
            for (int i = 0; i < TM; i++)
                a_frag[i] = As[thread_row + i][k];
            for (int j = 0; j < TN; j++)
                b_frag[j] = Bs[k][thread_col + j];

            // 寄存器上做外积
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    c_frag[i][j] += a_frag[i] * b_frag[j];
        }
        __syncthreads();
    }

    // 写回 Global Memory
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++)
            C[(by * BM + thread_row + i) * N + bx * BN + thread_col + j] = c_frag[i][j];
}





