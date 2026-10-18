#include "sgemm/SGEMM.cuh"

namespace CUDAkernels {

void instantiated_sgemm_block_tiling(cudaStream_t s, const float *A, const float *B, float *C, int M, int N, int K) {

    constexpr int BM = 64; // 64
    constexpr int BN = 64; // 64
    constexpr int BK = 32; // 32
    constexpr int BLOCK_SIZE = 256; // 256
    // 每线程寄存器数量
    // 


    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    

    sgemm_block_tiling<BM, BN, BK, BLOCK_SIZE> <<<grid, block, 0, s>>>(A, B, C, M, N, K);
}

} // namespace CUDAkernels


