#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cuda/std/cmath>

__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if(workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}


void f() {
    float* nu1 = nullptr;
    float* nu2 = nullptr;
    float* su = nullptr;

    int threadsPerBlock = 256;
    int vecLen = 1024;
    int blockNum = (vecLen + threadsPerBlock - 1) / threadsPerBlock;
    // //um 统一内存
    // cudaMallocManaged(&nu1, blockNum * threadsPerBlock);
    // cudaMallocManaged(&nu2, blockNum * threadsPerBlock);
    // cudaMallocManaged(&su, blockNum * threadsPerBlock);
    // // initialize nu1 nu2
    // // ...

    //explicit memory managed
    float* devNu1 = nullptr;
    float* devNu2 = nullptr;
    float* devSu = nullptr;
    cudaMallocHost(&nu1, blockNum * threadsPerBlock);
    cudaMallocHost(&nu2, blockNum * threadsPerBlock);
    cudaMallocHost(&su, blockNum * threadsPerBlock);
    // initialize nu1 nu2
    // ...

    cudaMalloc(&devNu1, blockNum * threadsPerBlock);
    cudaMalloc(&devNu2, blockNum * threadsPerBlock);
    cudaMalloc(&devSu, blockNum * threadsPerBlock);

    cudaMemcpy(devNu1, nu1, blockNum * threadsPerBlock, cudaMemcpyHostToDevice);
    cudaMemcpy(devNu2, nu2, blockNum * threadsPerBlock, cudaMemcpyHostToDevice);
    cudaMemset(devSu, 0, blockNum * threadsPerBlock);



    dim3 grid{(unsigned)blockNum, 1, 1};
    dim3 block{(unsigned)threadsPerBlock, 1, 1};
    vecAdd<<<grid, block>>>(nu1, nu2, su, vecLen);

    cudaDeviceSynchronize();


    //explicit memory managed
    cudaMemcpy(&su, &devSu, blockNum * threadsPerBlock, cudaMemcpyDeviceToHost);

    cudaFree(devNu1);
    cudaFree(devNu2);
    cudaFree(devSu);
    cudaFreeHost(nu1);
    cudaFreeHost(nu2);
    cudaFreeHost(su);

    //um 统一内存
    cudaFree(nu1);
    cudaFree(nu2);
    cudaFree(su);
    
}