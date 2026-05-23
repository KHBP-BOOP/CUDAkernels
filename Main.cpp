#include "vecAdd.cuh"
#include <iostream>


int main() {
#ifdef MINI_VERSION_CPP

    std::cout << "success" << std::endl;
#else
    std::cout << "fail" << std::endl;

#endif


    //testVecAdd();



    // int minGridSize = 0;
    // int blockSize = 0;
    // cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, testFunc, 0, 0);
    // int gridSize = (1024 + blockSize - 1) / blockSize;
}