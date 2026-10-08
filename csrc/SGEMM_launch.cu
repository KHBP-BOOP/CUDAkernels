

#include "sgemm/SGEMM_v1.cuh"

namespace CUDAkernels {

void instantiated_sgemm_block_tiling_v1(cudaStream_t s, const float *A, const float *B, float *C, int M, int N, int K) {

    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 4;
    constexpr int BLOCK_SIZE = 256;

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);


    sgemm_block_tiling_v1<BM, BN, BK, BLOCK_SIZE> <<<grid, block, 0, s>>>(A, B, C, M, N, K);
}

} // namespace CUDAkernels


