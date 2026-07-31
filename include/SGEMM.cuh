#pragma once

#include <cuda_runtime_api.h>

// SGEMM 主机端启动封装（核函数模板定义与实例化见 src/SGEMM.cu）
// 实例化模板参数: BM=128, BN=128, BK=8, BLOCK_SIZE=256, Wx=8, Wy=4, TM=8, TN=8
void launch_sgemm_thread_tiling(const float *A, const float *B, float *C,
                                int M, int N, int K, dim3 grid, dim3 block);

// 主机端测试入口（定义见 src/testSGEMM.cu）
void testSGEMM();
