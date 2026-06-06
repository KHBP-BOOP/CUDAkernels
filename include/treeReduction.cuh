#pragma once

#include <cuda_runtime_api.h>

__global__ void tree_reduction(float* input, float* output, int size);

template <int BLOCK_SIZE>
__global__ void tree_reduction(float* input, float* output, int n);

void testTreeReduction();