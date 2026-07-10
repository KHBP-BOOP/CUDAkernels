# SGEMM
# tensor core

***GEMM 优化的本质是用寄存器和共享内存（Shared Memory）挡住对全局内存（Global Memory）的访问。*** 

C = A @ B

$A \in \mathbb{R}^{M \times K}$

$B \in \mathbb{R}^{K \times N}$

$C \in \mathbb{R}^{M \times N}$

### 性能指标

不考虑落地工程中的端到端耗时与延迟、鲁棒性、每瓦特吞吐量等因素，算力利用率、带宽利用率是以结果为导向的衡量指标；算术强度、全局内存访问率、L1L2缓存命中率是以过程为导向的衡量指标。

![alt text](image.png)


### tiling

分块思想贯穿始终

***每下降一个内存层次，就对应线程层次的一层分块。***

## Naive版本

SGEMM 计算强度I =

2 * M * N * K / 4 * 2 * M * N * K = 0.25FLOPs/Byte


## version 1

thread block级tiling
矩阵C划分为BM×BN 的分块，每个 Thread Block 负责一块

SGEMM 计算强度I =

2 * BM * BN * BK / 4 * (BM * BK + BK * BN)

BM = BN = 64  ->  I == 16 FLOPS/Byte
BM = BN = 128  ->  I == 32 FLOPS/Byte


二维grid 一维block256




根据C矩阵M*N的尺寸分配线程数量与尺寸（），r0用于A、C矩阵行索引，c0用于B、C矩阵列索引，A矩阵列索引、B矩阵行索引借助K-Loop中的循环变量k

一个tileA，128 * 8，1024个元素，由一维线程块负责，包含256个thread，重排为32 * 8，使用跨步循环实现一个block覆盖tileA所有元素；

一个tileB，8 * 128，1024个元素，由一维线程块负责，包含256个thread，重排为8 * 32，使用跨步循环实现一个block覆盖tileB所有元素；



用 a_thread_x/y 和 b_thread_x/y 的线程重排索引，目的是为了实现合并内存访问（Coalesced Access）

一个block负责一排As和一列Bs的矩乘，一个线程沿完整K维度计算，通过循环实现跨步覆盖，并在simt架构的前提下借助多个线程实现局部覆盖（单排As与单列Bs）
tileAB数据载入SMEM并同步后，开始并行计算。单个线程在Tm、Tn维度的跨步循环过程中，于Ct中累加结果，外层k维度的循环结束后，该线程负责的多行多列的计算在BK维度上完成；随后进行block级同步；外层K-LOOP结束后，该线程负责的多行多列的计算在K维度上完成；
Ct为该thread负责的多排多列的最终矩乘结果，

tileC分为64个格，一个线程跨步计算每个格中的一个元素，共64个元素，由寄存器中的Ct变量存储。

？？？？：
1616 thread借助循环处理128128个元素，目的？


为什么block采用一维？为什么256个thread？
瓦片A 瓦片B 尺寸设定：
BK如何确定




