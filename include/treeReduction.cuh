#pragma once

#include <cuda_runtime_api.h>

__global__ void tree_reduction(float* input, float* output, int size);

void testTreeReduction();