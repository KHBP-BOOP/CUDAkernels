
v1 --- v1.1（reg）
|      |
v1.2   v1.1.1（reg、合并访存）
（合并访存、LDGSTS）
# tail effect






# triage

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

- 首先将单发版的load As Bs改为跨步循环版，以便后续更改启动配置并增强鲁棒性，其他不变  
实测运行519808 ns，在预期范围内

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
LSU高占用，

## 调整
引入寄存器数组，以提高数据复用率，从而减少指令数
v1.3性能无明显变化，从v1.2的 548544 ns变为 548

## 分析
说明此前编译器已经进行了对应优化。

注意到
SMEM加载至寄存器的指令LDS 异常多，存储至SMEM的指令STS 异常少
GMEM加载至寄存器的指令LDG 异常少
分析源码，发现LDS指令大多在As、Bs中数据被load至寄存器时被发射。

16*16的tileC中，As中数据被load至寄存器afrag时，256个线程均执行了LDS，但实际上只需**lane % 16 == 0的线程**的LDS指令需要被发射，其他线程进行warp内广播即可



## 调整










## 分析

检查遗漏信息  

两大子系统吞吐双双低于50%，判断延迟受限

No Eligible [%]	49.32
One or More Eligible [%]	50.68
说明在活跃周期的50%中不发指令
Active Warps Per Scheduler [warp]	3.39
接近每scheduler的理论warps数 4
Eligible Warps Per Scheduler [warp]	1.44
eligible warps占active warps的42.48%
说明eligible warps占比过少
Avg. Active Threads Per Warp	32
不存在分歧

eligible warps占比42.48% 导致了在50%的活跃周期中不发指令，这通常意味着内存延迟隐藏不佳，需要检查Warp State 图表、Warp Stall Sampling (Not-issued Samples)图表

**问题**：
内存延迟隐藏不佳

经排查，未发射指令情况下，warp被stalled的原因中Stall Long Scoreboard占比平均94%以上，且集中在*float4向量化的数据在写入As、Bs时*。
float4向量化的数据在写入As、Bs时，对GMEM的访问合并情况不佳；对于As，每运输BK个fp32就被分为一个事务（本次配置BK=8）；对于Bs，每运输BN个fp32就被分为一个事务（本次配置BN=128），合并情况较优于As但仍很差。这一步过慢的速度导致warps花费太多时间*等待GMEM中的数据进入L1TEX即SMEM*。这对应了Memory Throughput Breakdown图表中的L1TEX数据管线、输入输出百分比断档领先的异常现象




## 调整

- 优化访问GMEM的方式提高合并访存效率：

先在v1.1上修改 - v1.1.1
负优化

再在v1上修改（同时引入了v1.1的跨步load As、Bs，tile C线程重拍） - v1.2



## 分析

Registers Per Thread [register/thread]	125
每线程寄存器数过大，执行Ct[i] += As[row][p] * Bs[p][col];一步时寄存器数量总是明显更大
压缩tileA、B尺寸



