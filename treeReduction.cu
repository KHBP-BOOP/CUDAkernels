#include <cuda_runtime_api.h>

__global__ void tree_reduction(float* input, float* output, int size) {
    extern __shared__ float shMem[];
    int tdx = threadIdx.x;
    int gdx = blockDim.x * blockIdx.x + tdx;

    shMem[tdx] = gdx < size ? input[gdx] : 0.0f;
    __syncthreads();

    for (int step = blockDim.x / 2; step > 0; step /= 2) {
        if (tdx < step) { //仅需数组的前半部分累加
            shMem[tdx] += shMem[tdx + step];
        }
        __syncthreads(); //保证下一次运算读取的是更新后的数值
    }


    if (tdx == 0) {
        output[blockDim.x] = shMem[0];
    }

}