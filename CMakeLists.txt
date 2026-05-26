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

set(SCOURSE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vecAdd.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vectorDotProduct.cu
)

#生成可执行文件
add_executable(EXEFILE ${SCOURSE})

target_include_directories(EXEFILE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

