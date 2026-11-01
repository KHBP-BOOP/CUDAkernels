# 开发环境

GPU架构  Ada Lovelace
GPU型号 NVIDIA GeForce RTX 4060
计算能力 8.9
GPU驱动版本 610.47

PyTorch版本 2.13.0
PyTorch使用的CUDA runtime版本 13.0
CUDA Toolkit 13.2.78

GCC 13.3.0
基础c运行库 glibc
CPU架构 x86_64
os内核 linux(WSL2)
c++标准 c++20

# Pytorch配置详情

- GCC 13.3
- C++ Version: 202002
- CPU capability usage: AVX2
- CUDA Runtime 13.0
- NVCC architecture flags: -gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_90,code=sm_90;-gencode;arch=compute_100,code=sm_100;-gencode;arch=compute_120,code=sm_120
- CuDNN 92.0  (built against CUDA 13.2)
- Intel(R) oneAPI Math Kernel Library Version 2024.2-Product Build 20240605 for Intel(R) 64 architecture applications
- Intel(R) MKL-DNN v3.12.0 (Git Hash 80afa71049cd69a3df32adcccb623b12cd7baa22)
- OpenMP 201511 (a.k.a. OpenMP 4.5)
- LAPACK is enabled (usually provided by MKL)
- NNPACK is enabled
- Magma 2.6.1

# 性能测试
使用nsight compute时的分析过程均会锁定GPU时钟频率
平均绝对执行时间均为10次独立运行的平均值

