#pragma once

#include <cuda_runtime_api.h>

void testVectorDotProduct();

__global__ void vectorDotProduct(const float* v1, const float* v2, float* ptrToDevSum, size_t size);
