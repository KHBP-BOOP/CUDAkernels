# v1

中间版本
v1.1 load阶段加入跨步循环（可在编译期内展开），设定BM = BN = 64，BK = 32，BZ = 256
v1.2 v1.2基础上加入寄存器数组frag
v1.3 线程读取向量化的As、Bs，
XOR swizzle，AsBs的存储显式规定为16字节对齐
v1.4 pipeline memcpy_async

最终版本

## 分析

首先判断工作负载是否足够高
由于Waves per SM 值为 1.33
不是整数，存在tail effect（v1中未解决的问题）
且
Block Limit Registers [block]	2
Block Limit Shared Mem [block]	7
Block Limit Warps [block]	6
Block Limit SM [block]	24
得知占用率的理论上限被每thread的寄存器数量限制（v1中未解决的问题）
Registers Per Thread [register/thread]	125

**问题**：
每线程占用寄存器数量过多，进而导致每sm上驻留的CTA数量过少，warp随之过少，occupancy理论值过低；
同时，由于CTA数量不是SM整数倍出现tail effect：
Grid Size 64，且SM 每GPU有24个，说明SM上经过了一个完整wave（每SM两个CTA，共48个CTA）、一个partial wave（仅16个SM上有一个CTA）

## 调整

- 将单发版的load As Bs改为跨步循环版

- 改BLOCK_SIZE、BM、BN、BK
首先1536 / BZ<=24	由于硬件条件限制 BZ==64 128 256 512 1024；
BLOCK_SIZE值取256以上时，Block Limit Warps [block] 将小于3，
BLOCK_SIZE值取128以下时，BM、BN被迫变小，且：
tileA、B的传输在SMEM和GMEM中发生，故根据不等式BM应等于BN且二者应尽可能大，直到相应步骤的算术强度大于DRAM的拐点横坐标值
故BZ取值暂定128/256

BM = BN = 256或128时，即使BLOCK_SIZE=256，单线程的寄存器数量也会分别暴涨至64、128，严重降低occupancy
若BK = 32，load As时每运输一排数据就对应一个128B事务，过大的BK如64可能导致每线程占用过多SMEM、STS.128时跨步循环次数过多
经实测，BM = BN = 64，BK = 32，BZ = 256运行时间最短，故暂定上述配置

v1.1出现负优化，从v1的 516704 ns 变为 548544 ns，但计算吞吐、内存吞吐接近90%、每线程寄存器降至64、实际occupancy 58.47%


## 分析
High instruction throughput + high LSU writeback
LSU高占用，指令数过多

注意到
SMEM加载至寄存器的指令LDS 异常多，存储至SMEM的指令STS 异常少
GMEM加载至寄存器的指令LDG 异常少
分析源码，发现LDS指令大多在As、Bs中数据被load至寄存器时被发射。

## 调整
引入寄存器数组，以提高数据复用率，从而减少指令数
v1.2性能无明显变化，从v1.1的 548544 ns变为 549312 ns


## 分析
说明此前编译器已经进行了对应优化。
展开细节分析时，发现：
16*16的tileC中，As中数据被load至寄存器afrag时，256个线程均执行了LDS，但实际上可以通过控制循环结构将大量LDS.32指令转换为较少的LDS.128，即先float4化，再运输，直接减少LDS指令数量

## 调整
v1.3出现正优化，从v1.2的 549312 ns 提升到 433920 ns


## 分析
- L1TEX内部LSU数据管线占用率偏高：requests/wavefronts大致为1/3，LDS的访问模式可能不佳
STS指令的requests/wavefronts == 1/4
二者访问方式可能都不佳
定位至float4数据从寄存器存入As、Bs过程

- Stall Short Scoreboard、Stall MIO Throttle、Stall Long Scoreboard均异常偏高
long定位于load As、Bs的STS.128，ta们都依赖紧邻的LDG.E.128返回的数据
short定位于在Ct中进行FFMA运算时，前面有5条LDS.128

## 调整
- pipeline + memcpy_async

> 如果所有warp都先运输再计算，LDG会导致大量的阻塞。考虑解耦拷贝与计算，并使二者能够并行执行
可以让一部分warp只负责拷贝、另外的warp只负责计算，解耦两个操作；同时将拷贝方式改为异步拷贝，实现拷贝操作与计算操作能够同时进行
依靠其中的LDGSTS指令，不仅能隐藏大量延迟，还能提高传输速度、减少寄存器使用
把GMEM中的待输入数据先从逻辑上分为num_batches个批次的小部分数据，再逐批次地运输

- Ct写回GMEM过程float4化


v1.4出现正优化，从433920 ns 提升至 431552 ns


## 分析
- 提升幅度较小

- 寄存器执行计算时的short scoreboard成为阻塞主因  
FFMA指令执行时可能耗费了许多时间等待LDS的执行

- Tm = Tn = 8时，FFMA/LDS比值最大，当前Tm = Tn = 4

可能是启动配置问题

## 调整
1. BZ=128
可取 BM=128 BN=64 BK
此时C_BLOCK_X=BN/Tn=8  C_BLOCK_Y=BM/Tm=16

BK  duration(ns)
32  387040
16  381728
8   422112
暂取BK=16

v1.4提升至382272 ns

C_B_X=4?
507734 ns


2. BZ=256
可取 BM=128 BN=128 BK
962304 ns




