#include <cuda_runtime_api.h>

__device__ float warpReduceSum(float val) {
    for (int step = 16; step > 0; step >>= 1) {
        //valOfThread no longer represents the value of thread
        val += __shfl_down_sync(0xFFFFFFFF, val, step);
    }
    return val;
}


__global__ void blockReduceSum(float* input, float* output, int length) {
    int gdx = blockDim.x * blockIdx.x + threadIdx.x;

    float valOfThread = gdx < length ? input[gdx] : 0.0f;

    //warp reduce sum
    valOfThread = warpReduceSum(valOfThread);

    


    extern __shared__ float numInLane0[]; //sizeof(float) * blockDim.x / 32
    if (threadIdx.x % 32 == 0) {
        numInLane0[threadIdx.x / 32] = valOfThread;
    }

    __syncthreads();

    valOfThread = threadIdx.x < (blockDim.x / 32) ? numInLane0[threadIdx.x] : 0.0f;
    valOfThread = warpReduceSum(valOfThread);

    if (threadIdx.x == 0) {
        output[blockIdx.x] = valOfThread;
    }
}