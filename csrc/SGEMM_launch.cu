#include "sgemm/SGEMM.cuh"

namespace CUDAkernels {

void instantiated_sgemm_block_tiling(cudaStream_t s, const float *A, const float *B, float *C, int M, int N, int K) {

    constexpr int BM = 128; // 128
    constexpr int BN = 64; // 64
    constexpr int BK = 16; // 16
    constexpr int BLOCK_SIZE = 128; // 128
    constexpr int numStages = 2;


    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(BLOCK_SIZE);

    sgemm_block_tiling<BM, BN, BK, BLOCK_SIZE, numStages> <<<grid, block, 0, s>>>(A, B, C, M, N, K);
}

} // namespace CUDAkernels


