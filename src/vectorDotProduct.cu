#include "vectorDotProduct.cuh"

#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "error occurs in %s of %d line: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0) \

__global__ void vectorDotProduct(const float* v1, const float* v2, float* ptrToDevSum, int size) {
    extern __shared__ float sharedArray[];
    int tdx = threadIdx.x;
    int gdx = blockDim.x * blockIdx.x + threadIdx.x;

    float singleProduct = 0.0f;
    while (gdx < size) {
        singleProduct += v1[gdx] * v2[gdx];
        gdx += blockDim.x * gridDim.x;
    }
    // !!! gdx no longer represents a global index


    sharedArray[tdx] = singleProduct;
    __syncthreads();

    for (int step = tdx / 2; step > 0; step /= 2) {
        if (tdx < step) {
            sharedArray[tdx] += sharedArray[tdx + step];
        }
        __syncthreads();
    }

    if (tdx == 0) {
        *ptrToDevSum += sharedArray[tdx];
    }
}



void testVectorDotProduct() {
    float* vec1 = nullptr;
    float* vec2 = nullptr;
    float* ptrToRes = nullptr;

    int minGridSize = 0;
    int blockSize = 0;
    CUDA_CHECK(  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, reinterpret_cast<const void*>(vectorDotProduct), 0, 0);  );
    
    int vecLen = 1 << 24;
    int gridSize = (vecLen - 1) / blockSize + 1;


    cudaMallocHost(&vec1, vecLen * sizeof(float));
    cudaMallocHost(&vec2, vecLen * sizeof(float));
    cudaMallocHost(&ptrToRes, sizeof(float));

    //initialize
    for (int i = 0; i < vecLen; ++i) {
        vec1[i] = sinf(i * 0.0001f);
        vec2[i] = cosf(i * 0.0001f);
    }

    float* devVec1 = nullptr;
    float* devVec2 = nullptr;
    float* ptrToDevSum = nullptr;
    cudaMalloc(&devVec1, vecLen * sizeof(float));
    cudaMalloc(&devVec2, vecLen * sizeof(float));
    cudaMalloc(&ptrToDevSum, sizeof(float));




    
    cudaEvent_t start, end, kernelStart, kernelEnd;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelEnd);

    cudaEventRecord(start, 0);

    cudaMemcpy(devVec1, vec1, vecLen * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devVec2, vec2, vecLen * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(ptrToDevSum, 0.0f, sizeof(float));

    cudaEventRecord(kernelStart, 0);
    vectorDotProduct<<<gridSize, blockSize, blockSize * sizeof(float)>>>(devVec1, devVec2, ptrToDevSum, vecLen);
    cudaEventRecord(kernelEnd, 0);

    cudaError_t err = cudaGetLastError();
    cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err) << std::endl;
    }

    cudaMemcpy(ptrToRes, ptrToDevSum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    float durationTime = 0.0f, calculateTime = 0.0f;
    cudaEventElapsedTime(&durationTime, start, end);
    cudaEventElapsedTime(&calculateTime, kernelStart, kernelEnd);
    std::cout << "lasting: " << durationTime << std::endl;
    std::cout << "calculating: " << calculateTime << std::endl;
    std::cout << "result: " << *ptrToRes << std::endl;



    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaEventDestroy(kernelStart);
    cudaEventDestroy(kernelEnd);
    cudaFreeHost(vec1);
    cudaFreeHost(vec2);
    cudaFreeHost(ptrToRes);
    cudaFree(devVec1);
    cudaFree(devVec2);
    cudaFree(ptrToDevSum);


}