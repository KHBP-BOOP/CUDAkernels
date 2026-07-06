# SGEMM

C = A @ B

$A \in \mathbb{R}^{M \times K}$

$B \in \mathbb{R}^{K \times N}$

$C \in \mathbb{R}^{M \times N}$

#### 性能指标




### Naive版本

SGEMM 计算强度I =

2 * M * N * K / 4 * 2 * M * N * K = 0.25FLOPs/Byte


### version 1

thread block级tiling

SGEMM 计算强度I =

2 * BM * BN * BK / 4 * (BM * BK + BK * BN)

BM = BN = 64  ->  I == 16 FLOPS/Byte
BM = BN = 128  ->  I == 32 FLOPS/Byte

瓦片A 瓦片B 尺寸设定：




