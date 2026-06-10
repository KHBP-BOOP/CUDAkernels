cmake_minimum_required(VERSION 3.28) #设置CMake最小版本

project (PracticeForCUDA_cmake) #设置工程名

enable_language(CXX)
enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 89)

#set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE release)
endif()

string(APPEND CMAKE_CUDA_FLAGS " -lineinfo -Xptxas=-v")



#生成可执行文件
add_executable(EXEFILE
    src/Main.cpp
    src/vecAdd.cu
    src/vectorDotProduct.cu
    src/DEVICE.cu
    src/blockReduceSum.cu
    src/treeReduction.cu
)

set_source_files_properties(src/blockReduceSum.cu PROPERTIES COMPILE_FLAGS "-g -lineinfo")

find_program(NCU_EXECUTABLE ncu)
if(NCU_EXECUTABLE)
  add_custom_target(ncu_profile
    COMMAND ${NCU_EXECUTABLE} --set full --kernel-name regex:blockReduceSum
            --launch-skip 0 --launch-count 1 $<TARGET_FILE:EXEFILE>
    DEPENDS EXEFILE
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Run Nsight Compute (ncu) for kernel blockReduceSum"
  )
endif()


target_include_directories(EXEFILE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

