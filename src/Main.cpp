#include "vecAdd.cuh"
#include "vectorDotProduct.cuh"
#include "treeReduction.cuh"
#include "SGEMM.cuh"

#include <iostream>


int main() {



    //testVecAdd();
    //testVectorDotProduct();
    //testTreeReduction();
    testSGEMM();




    // int minGridSize = 0;
    // int blockSize = 0;
    // cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, testFunc, 0, 0);
    // int gridSize = (1024 + blockSize - 1) / blockSize;
}