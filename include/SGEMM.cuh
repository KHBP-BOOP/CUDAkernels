#pragma once

#include <cuda_runtime_api.h>


// SGEMM 主机端启动封装（核函数模板定义与实例化见 src/SGEMM.cu）
// 实例化模板参数: BM=128, BN=128, BK=8, BLOCK_SIZE=256, Wx=8, Wy=4, TM=8, TN=8
// grid/block 由各启动封装根据自身 tile 尺寸在内部计算，调用方只需给出问题规模
extern void launch_sgemm_thread_tiling_v3(const float *A, const float *B, float *C,
                                int M, int N, int K);

extern void launch_sgemm_thread_tiling_v4(const float *A, const float *B, float *C,
                                int M, int N, int K);

extern void launch_sgemm_thread_tiling_v5(const float *A, const float *B, float *C,
                                int M, int N, int K);

extern void launch_sgemm_thread_tiling_v1(const float *A, const float *B, float *C,
                                int M, int N, int K);


// 主机端测试入口（定义见 src/testSGEMM.cu）
void testSGEMM();
