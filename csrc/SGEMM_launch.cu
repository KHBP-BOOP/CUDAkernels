#include "sgemm/SGEMM.cuh"

namespace CUDAkernels {

void instantiated_sm89_fp32_gemm(cudaStream_t s, const float *A, const float *B, float *C, int M, int N, int K) {

    constexpr int BM_small = 128; // 128
    constexpr int BM_large = 64; // 64
    constexpr int BN = 64; // 64
    constexpr int BK = 16; // 16
    constexpr int BLOCK_SIZE = 128; // 128
    constexpr int numStages = 2;

    dim3 block(BLOCK_SIZE);

    //TODO : 更细粒度的分配策略
    if (M >= 2048 && N >= 2048 && K >= 2048) { // 大矩阵

        dim3 grid((N + BN - 1) / BN, (M + BM_small - 1) / BM_small);

        sm89_fp32_gemm<BM_small, BN, BK, BLOCK_SIZE, numStages>
            <<<grid, block, 0, s>>>(A, B, C, M, N, K);
    }

    else { // 小矩阵

        // 仅M、N、K都可被整除时，才能整块跳过所有边界判断
        const bool FULL_TILE = (M % BM_large == 0) && (N % BN == 0) && (K % BK == 0);
        dim3 grid((N + BN - 1) / BN, (M + BM_large - 1) / BM_large);

        if (FULL_TILE) {
            sm89_fp32_gemm<BM_large, BN, BK, BLOCK_SIZE, numStages, true>
                <<<grid, block, 0, s>>>(A, B, C, M, N, K);
        }

        else {
            sm89_fp32_gemm<BM_large, BN, BK, BLOCK_SIZE, numStages, false>
                <<<grid, block, 0, s>>>(A, B, C, M, N, K);
        }        
    }
}

} // namespace CUDAkernels


