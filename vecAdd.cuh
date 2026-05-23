#pragma once

#include <cuda_runtime_api.h>


__global__ void vecAdd(float* A, float* B, float* C, int vectorLength);

void testVecAdd();

