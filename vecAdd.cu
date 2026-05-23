#include "vecAdd.cuh"

#include <iostream>
#include <cmath>
#include <cuda_runtime_api.h>


__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
    int step = blockDim.x * gridDim.x;

    for (int i = workIndex; i < vectorLength; i += step) {
        C[workIndex] = A[workIndex] + B[workIndex];
    }

}


void testVecAdd() {
    float* nu1 = nullptr;
    float* nu2 = nullptr;
    float* su = nullptr;
    
    int minGridSize = 0;
    int blockSize = 0;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, vecAdd, 0, 0);

    int vecLen = 1024;
    int gridSize = (vecLen + blockSize - 1) / blockSize;
    // //um 统一内存
    // cudaMallocManaged(&nu1, gridSize * blockSize * sizeof(float));
    // cudaMallocManaged(&nu2, gridSize * blockSize * sizeof(float));
    // cudaMallocManaged(&su, gridSize * blockSize * sizeof(float));
    // // initialize nu1 nu2
    // // ...

    //explicit memory managed
    float* devNu1 = nullptr;
    float* devNu2 = nullptr;
    float* devSu = nullptr;
    cudaMallocHost(&nu1, gridSize * blockSize * sizeof(float));
    cudaMallocHost(&nu2, gridSize * blockSize * sizeof(float));
    cudaMallocHost(&su, gridSize * blockSize * sizeof(float));
    // initialize nu1 nu2
    const int size = (1 << 24) * sizeof(float);
    for (int i = 0; i < size; ++i) {
        devNu1[i] = sinf(i * 0.0001f);
        devNu2[i] = cosf(i * 0.0001f);
    }

    cudaMalloc(&devNu1, gridSize * blockSize * sizeof(float));
    cudaMalloc(&devNu2, gridSize * blockSize * sizeof(float));
    cudaMalloc(&devSu, gridSize * blockSize * sizeof(float));

    cudaMemcpy(devNu1, nu1, gridSize * blockSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devNu2, nu2, gridSize * blockSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(devSu, 0, gridSize * blockSize * sizeof(float));



    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start, 0);
    vecAdd<<<gridSize, blockSize>>>(nu1, nu2, su, vecLen);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float durationTime = 0.0f;
    cudaEventElapsedTime(&durationTime, start, end);
    std::cout << "lasting: " << durationTime << std::endl;

    cudaDeviceSynchronize();


    //explicit memory managed
    cudaMemcpy(&su, &devSu, gridSize * blockSize * sizeof(float), cudaMemcpyDeviceToHost);

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