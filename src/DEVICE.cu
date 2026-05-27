__device__ void warp_broadcast(float* n1) {

    float num = 3.0f * threadIdx.x;

    float broadcastNum = __shfl_sync(0xFFFFFFFF, num, 1);



}