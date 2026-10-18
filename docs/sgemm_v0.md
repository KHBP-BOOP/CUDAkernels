

```
// v0
template <int BM, int BN, int BK, int BLOCK_SIZE>
__global__ void sgemm_block_tiling_v1(const float* A, const float* B, float* C,
                                   int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int r0 = blockIdx.y * BM;
    int c0 = blockIdx.x * BN;
    int tid = threadIdx.x;

    // 加载 tileA 时的线程重排
    constexpr int A_BLOCK_X = BK;  // = 4
    // constexpr int A_BLOCK_Y = BLOCK_SIZE / A_BLOCK_X;  // = 64
    int a_thread_x = tid % A_BLOCK_X; // 0 ~ 3
    int a_thread_y = tid / A_BLOCK_X; // 0 ~ 63

    // 加载 tileB 时的线程重排
    constexpr int B_BLOCK_X = BN;  // = 64
    // constexpr int B_BLOCK_Y = BLOCK_SIZE / B_BLOCK_X;  // = 4
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
```