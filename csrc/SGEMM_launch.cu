#include "sgemm/SGEMM.cuh"

namespace CUDAkernels {

void instantiated_sm89_fp32_gemm(cudaStream_t s, const float *A, const float *B, float *C, int M, int N, int K) {

    constexpr int BK = 16; // 16
    constexpr int BN = 64; // 64
    constexpr int numStages = 2;

    // TODO : 更细粒度的分配策略
    // BM、BLOCK_SIZE分类讨论
    if (M < 1024 && N < 1024 && K < 1024) { // 小矩阵

        constexpr int BLOCK_SIZE = 128;
        constexpr int BM = 128; // 128
        dim3 block(BLOCK_SIZE);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

        sm89_fp32_gemm<BM, BN, BK, BLOCK_SIZE, numStages>
            <<<grid, block, 0, s>>>(A, B, C, M, N, K);
    }

    else { // 大矩阵

        constexpr int BLOCK_SIZE = 64; // 128
        constexpr int BM = 64; // 64
        dim3 block(BLOCK_SIZE);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

        // 仅M、N、K都可被整除时，才能整块跳过所有边界判断
        const bool FULL_TILE = (M % BM == 0) && (N % BN == 0) && (K % BK == 0);        
        if (FULL_TILE) {
            sm89_fp32_gemm<BM, BN, BK, BLOCK_SIZE, numStages, true>
                <<<grid, block, 0, s>>>(A, B, C, M, N, K);
        }

        else {
            sm89_fp32_gemm<BM, BN, BK, BLOCK_SIZE, numStages, false>
                <<<grid, block, 0, s>>>(A, B, C, M, N, K);
        }        
    }
}

} // namespace CUDAkernels


