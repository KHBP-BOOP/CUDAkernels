
#include "treeReduction.cuh"

#include <cstdio>
#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "error occurs in %d line of %s : %s", __LINE__, __FILE__, cudaGetLastError(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)


__device__ void reduceSumInLastWarp(int val, int tdx) {
    //虽然同一个 Warp 内的线程是同时发射指令的（SIMT 同步），但指令同步不等于数据可见。如果数据被锁死在私有寄存器里，别的线程在物理上就是读不到
    //在同一个 Warp 内部，32 个线程天生单指令多线程（SIMT）同步执行

    // shMem[tdx] += shMem[tdx + 32];
    // shMem[tdx] += shMem[tdx + 16];
    // shMem[tdx] += shMem[tdx + 8];
    // shMem[tdx] += shMem[tdx + 4];
    // shMem[tdx] += shMem[tdx + 2];
    // shMem[tdx] += shMem[tdx + 1];

    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
}

//grid应配置为(n + blockDim.x * 2 - 1) / (blockDim.x * 2)
//保证每个block均有不为0的数据且不丢失数据
__global__ void tree_reduction(float* input, float* output, int size) {
    extern __shared__ float shMem[];
    int tdx = threadIdx.x;
    int gdx = 2 * blockDim.x * blockIdx.x + tdx;

    float val = 0.0f;
    if (gdx < size) {
        val += input[gdx];
    }
    if (gdx + blockDim.x < size) {
        val += input[gdx];
    }
    shMem[tdx] = val;
    __syncthreads();

    //step <= 32 时退出循环
    for (int step = blockDim.x / 2; step > 32; step /= 2) {
        if (tdx < step) { //仅需数组的前半部分累加
            shMem[tdx] += shMem[tdx + step];
        }
        __syncthreads(); //保证下一次运算读取的是更新后的数值
    }

    //最后一个warp内的规约求和
    if (tdx < 32) {
        int val = shMem[tdx];
        reduceSumInLastWarp(val, tdx);
    }


    if (tdx == 0) {
        output[blockIdx.x] = shMem[0];
    }

}


void testTreeReduction() {
    const int size = 1 << 27;
    const int threadsPerBlock = 256;
    // 每一个 Block 处理 threadsPerBlock 个元素
    const int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    size_t input_bytes = size * sizeof(float);
    size_t output_bytes = blocksPerGrid * sizeof(float);

    std::cout << "Allocating Host Memory..." << std::endl;
    // 2. 分配主机内存并初始化（全部设为 1.0f，方便最后验证正确性）
    std::vector<float> h_input(size, 1.0f);
    std::vector<float> h_output(blocksPerGrid, 0.0f);

    // 3. 分配设备（GPU）内存
    float *d_input, *d_output;
    cudaMalloc(&d_input, input_bytes);
    cudaMalloc(&d_output, output_bytes);

    // 4. 将输入数据从主机拷贝到设备
    cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice);

    // 5. 计算动态共享内存的大小（每个线程一个 float）
    size_t sharedMemSize = threadsPerBlock * sizeof(float);

    // 6. 创建 CUDA Event 用于高精度计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 【重要】预热 GPU (Warm-up)，消除驱动懒加载和显卡从省电模式唤醒的延迟
    tree_reduction<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input, d_output, size);
    cudaDeviceSynchronize();

    // 7. 循环执行多次，取平均时间以获得更稳定的带宽数据
    const int iterations = 100;
    std::cout << "Running kernel for " << iterations << " iterations..." << std::endl;
    
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; ++i) {
        tree_reduction<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input, d_output, size);
    }
    cudaEventRecord(stop, 0);
    
    // 等待 GPU 核心全部执行完毕
    cudaEventSynchronize(stop);

    // 计算总耗时与平均耗时
    float total_milliseconds = 0;
    cudaEventElapsedTime(&total_milliseconds, start, stop);
    float avg_milliseconds = total_milliseconds / iterations;

    // 8. 将结果拷贝回主机并进行数据校验
    cudaMemcpy(h_output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost);

    // 在 CPU 上对每个 block 的结果进行最后的规约
    float gpu_final_result = 0;
    for (int i = 0; i < blocksPerGrid; ++i) {
        gpu_final_result += h_output[i];
    }
    float expected_result = static_cast<float>(size); // 因为每个元素都是 1.0f

    // 9. 性能与带宽指标计算
    // 有效数据流转量：读取了整个 input 数组 + 写入了整个 output 数组
    double total_bytes_accessed = static_cast<double>(input_bytes + output_bytes);
    // 带宽计算公式: GB/s = (Bytes / 10^9) / (Seconds)
    double avg_seconds = avg_milliseconds / 1000.0;
    double bandwidth_gb_s = (total_bytes_accessed / 1e9) / avg_seconds;
    double peakBandWidth_gb_s = 256.0;

    // --- 打印测试报告 ---
    std::cout << "数据规模    : " << size << " 个 float (" << input_bytes / (1024.0 * 1024.0) << " MB)" << std::endl;
    std::cout << "Grid 配置   : " << blocksPerGrid << " Blocks, " << threadsPerBlock << " Threads/Block" << std::endl;
    std::cout << "结果验证    : " << (std::abs(gpu_final_result - expected_result) < 1e-4 ? "通过 (PASS)" : "失败 (FAIL)") << std::endl;
    std::cout << "GPU 计算结果: " << gpu_final_result << " (预期值: " << expected_result << ")" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "平均计算耗时: " << avg_milliseconds << " ms" << std::endl;
    std::cout << "有效内存带宽: " << bandwidth_gb_s << " GB/s" << std::endl;
    std::cout << "带宽占用率: " << 100 * bandwidth_gb_s / peakBandWidth_gb_s << " %" << std::endl;

    // 10. 释放资源
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);

}