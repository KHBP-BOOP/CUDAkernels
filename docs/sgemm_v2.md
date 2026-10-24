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
