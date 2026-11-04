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

- XOR swizzle 并 将AsBs的存储显式规定为16字节对齐

- 调整写回过程的布局，在一定程度上保证连续线程读取连续数据


v2.1出现正优化，绝对运行时间由 v1.4 382272 ns 提升至 360672 ns

## 分析
- Compute (SM) Throughput [%]	61.91
Memory Throughput [%]	43.39
可能受限于延迟过大

- stall原因中not-selected占比最大

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

## 分析
主因在于v1中内积计算时的short scoreboard stall

## 调整
- 考虑shuffle warp内广播，以减少LDS指令数；但同样需LSU参与且重构范围较大，暂不考虑

- load b-frag寄存器过程采用双缓冲方式进行
但注意到One or More Eligible [%]	70.95
指令发射率并不低
且ALU管线（地址计算，大概率对应swizzle部分）利用率与LSU管线接近、CBU管线（分支判断）仅次于FFMA、LSU、ALU
代码中每次storeAs、Bs均进行一次swizzle相关地址计算、大量用于边界判断的if语句

考虑先降低ALU、CBU管线利用率，再视情况考虑load b-frag过程的2 buffer处理
考虑根据M、N、K值分析边界判断的必要性；同时去除多余的和边界判断、地址计算相关的指令，减少每warp执行的指令数，直接性地缩短执行时间

对于单个线程而言 M, N, K, r0, c0, k0, tid是固定值，无需每次调用store_load_*时都算一遍
1. store As、Bs过程存在多余判断
分别对应`if (i < BM)`、`if (j < BK)`，此判断非必需
2. 解耦 处理full size 与 处理任意size ，提供两种解决方法：`store_tile_*` `store_tile_*_full`
3. swizzle地址计算
*v2.2出现正优化，执行时间 331520 ns*


stall short scoreboard比值大于1
考虑load b-frag过程的2 buffer处理

*v2.3  331712 ns*

stall short scoreboard比值仍为1.09
计算前有4条LDS.128
考虑pp循环中设计为先计算再预取

*执行时间、short scoreboard stall无变化*
这是因为计算前总会等待一次拷贝

active很多，eligible不足50%
fma lite成瓶颈

表明：对LDS的依赖关系无法解决，导致大量active warps无法进入eligible状态
40.62%的stall采样采集于内积计算，该步的细分stall原因中short scoreboard占比39.29%
提高指令流的平衡性，即访存与计算尽量交替进行
BK loop外的预取改为预取2个slot

*执行时间缩短至328ms*

考虑3 buffer

先存0，1槽位
算0，存2
算1，存0
算2，存1
算0，存2
...

*无明显提升 short scoreboard维持原值*

BZ设为64，此时Tm==Tn==8，提高访存比，减少指令数
*稳定在318144 ns*




