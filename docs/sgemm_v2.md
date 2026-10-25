# v2

## 分析
- 计算、内存吞吐均位于60%左右，延迟隐藏可能不佳
One or More Eligible [%]	66.73
指令发射效率不低

- Theoretical Occupancy [%]	25
Achieved Occupancy [%]	22.44
完整cycles warp停顿原因中仅short scoreboard远超baseline
对应内积计算+写入Ct过程

- 报告显示derived__memory_l2_theoretical_sectors_global_excessive 和  
derived__memory_l1_wavefronts_shared_excessive 较高，分别对应Bs写入b_frag过程和Ct
写回GMEM过程

Bs写入b_frag过程可能发生了bank conflict；写回过程访问模式可能不佳

col0 = c_thread_x * Tn，
每个thread读Tn个数据，每两个列索引相邻的线程所读取的数据相隔Tn个sizeof(float)字节；
Tn被设定为8，意味着每个线程从SMEM读取两次（这两次读取为串行执行），每次读取一个float4数据。
即发生了2-way bank conflict：每线程访问16B，但相邻线程访问的数据相隔32B，导致8个线程的访问落在了4个bank group（这里的group宽度为4个bank宽度）

写回过程与上述情景类似：每线程访问16B，但相邻线程访问的数据相隔32B，严重拉低了访问的合并度



## 调整
v2.1出现正优化，绝对运行时间由 v1.4 382272 ns 提升至 360672 ns

## 分析
- Compute (SM) Throughput [%]	61.91
Memory Throughput [%]	43.39
可能受限于延迟过大

- stall原因中not-selected占比最大

- LDS、LDGSTS指令对应的wavefronts/requests比例过大

- 理论occupancy仍低至25%
严重受限于每thread寄存器数量以及每block SMEM大小


## 调整
- 启动配置调整
`Tn * Tm * BZ === BM * BN`
 8    8    128    128  64

FFMA / LDS = 4 * Tm * Tn / (Tm + Tn)
考虑设定Tm = 4 Tn = 8，将a frag数组大小减半
同时改变Tm、Tn的尺寸定义逻辑，使得以下等式成立

`Tn * Tm * BZ === BM * BN`
 8    4    128    64   64

*调整后执行时间减慢，但occupancy 50%左右、内存计算吞吐均70%左右*

- 考虑寄存器数组存放float2

- 考虑split-k（k取2、3、4、5）

- 考虑shuffle warp内广播，以减少LDS指令数；但同样需LSU参与且重构范围较大，暂不考虑

- 去除多余的和边界判断、地址计算相关的指令
对于单个线程而言 M, N, K, r0, c0, k0, tid是固定值，无需每次调用store_load_*时都算一遍
1. store As、Bs过程存在多余判断
分别对应`if (i < BM)`、`if (j < BK)`，此判断非必需
2. 解耦 处理full size 与 处理任意size ，提供两种解决方法：`store_tile_*` `store_tile_*_full`

3. swizzle地址计算
*出现正优化，执行时间 331520 ns*

- load b-frag寄存器过程采用双缓冲方式进行

