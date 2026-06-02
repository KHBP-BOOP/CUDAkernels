#include "vecAdd.cuh"

#include <cmath>
#include <iostream>

#include <cuda_runtime_api.h>

#define CUDA_CHECK(call)\
do {\
    cudaError_t err = call;\
    if (err != cudaSuccess) {\
        fprintf(stderr, "CUDA error at %s:%d - %s\n",\
            __FILE__, __LINE__, cudaGetErrorString(err));\
        exit(EXIT_FAILURE);\
    }\
} while(0)\



__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
    int step = blockDim.x * gridDim.x;

    float4* vectorizedA = reinterpret_cast<float4*>(A);
    float4* vectorizedB = reinterpret_cast<float4*>(B);
    float4* vectorizedC = reinterpret_cast<float4*>(C);

    for (int i = workIndex; i < vectorLength / 4; i += step) {
        vectorizedC[i].x = vectorizedA[i].x + vectorizedB[i].x;
        vectorizedC[i].y = vectorizedA[i].y + vectorizedB[i].y;
        vectorizedC[i].z = vectorizedA[i].z + vectorizedB[i].z;
    }

    int tail_start = (vectorLength / 4) * 4;
    if (tail_start < vectorLength) {
        int tdx = threadIdx.x;
        C[tail_start + tdx] = A[tail_start + tdx] + B[tail_start + tdx];
    }

}


void testVecAdd() {
    float* nu1 = nullptr;
    float* nu2 = nullptr;
    float* su = nullptr;
    
    int minGridSize = 0;
    int blockSize = 0;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, vecAdd, 0, 0);

    int vecLen = 1 << 24;
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
    for (int i = 0; i < vecLen; ++i) {
        nu1[i] = sinf(i * 0.0001f);
        nu2[i] = cosf(i * 0.0001f);
    }

    cudaMalloc(&devNu1, gridSize * blockSize * sizeof(float));
    cudaMalloc(&devNu2, gridSize * blockSize * sizeof(float));
    cudaMalloc(&devSu, gridSize * blockSize * sizeof(float));

    cudaMemset(devSu, 0, gridSize * blockSize * sizeof(float));



    cudaEvent_t start, end, kernelStart, kernelEnd;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelEnd);

    //start to store on GPU
    cudaEventRecord(start, 0);

    cudaMemcpy(devNu1, nu1, gridSize * blockSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devNu2, nu2, gridSize * blockSize * sizeof(float), cudaMemcpyHostToDevice);

    //start to calculate
    cudaEventRecord(kernelStart, 0);
    vecAdd<<<gridSize, blockSize>>>(devNu1, devNu2, devSu, vecLen);
    CUDA_CHECK( cudaGetLastError() );
    //end to calculate
    cudaEventRecord(kernelEnd, 0);
    //cudaEventSynchronize(kernelEnd);
    
    //explicit memory managed
    cudaMemcpy(su, devSu, gridSize * blockSize * sizeof(float), cudaMemcpyDeviceToHost);

    //end to store on CPU
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    float durationTime = 0.0f, calculateTime = 0.0f;
    cudaEventElapsedTime(&durationTime, start, end);
    cudaEventElapsedTime(&calculateTime, kernelStart, kernelEnd);
    std::cout << "lasting: " << durationTime << std::endl;
    std::cout << "calculating: " << calculateTime << std::endl;



    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaEventDestroy(kernelStart);
    cudaEventDestroy(kernelEnd);
    cudaFree(devNu1);
    cudaFree(devNu2);
    cudaFree(devSu);
    cudaFreeHost(nu1);
    cudaFreeHost(nu2);
    cudaFreeHost(su);

    //um 统一内存
    //cudaFree(nu1);
    //cudaFree(nu2);
    //cudaFree(su);
    
}