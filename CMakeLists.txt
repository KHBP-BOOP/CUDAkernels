cmake_minimum_required(VERSION 3.28) #设置CMake最小版本

project(CUDAkernels_cmake) #设置工程名

find_package(CUDAToolkit REQUIRED)

enable_language(CXX)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 89)

set(CMAKE_BUILD_TYPE release)
# set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG" CACHE STRING "" FORCE)
# set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG" CACHE STRING "" FORCE)

set(TORCH_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test/.venv/lib64/python3.12/site-packages/torch/lib")
set(TORCH_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test/.venv/lib64/python3.12/site-packages/torch/include")






add_library(cudakernels_torch SHARED
    csrc/SGEMM_launch.cu
    bindings/torch_binding.cu
)

target_compile_definitions(cudakernels_torch PRIVATE USE_CUDA)

target_include_directories(cudakernels_torch
PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/sgemm/
PRIVATE
    ${TORCH_INCLUDE_DIR}/
)

# 与 torch 共用 libcudart.so.13
set_target_properties(cudakernels_torch PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

# 链 libtorch
target_link_libraries(cudakernels_torch
PRIVATE
    "${TORCH_LIB_DIR}/libtorch_cpu.so"     # 除下面一个外，全部符号都在这里
PRIVATE
    "${TORCH_LIB_DIR}/libtorch_cuda.so")   # torch_c10_cuda_check_msg







# 优先使用 CUDAToolkit 自带的 ncu（即使 CUDA bin 不在 PATH 中也能找到）
find_program(NCU_EXECUTABLE ncu HINTS "${CUDAToolkit_BIN_DIR}")

set(NCU_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/test/.venv/bin/python" CACHE FILEPATH "")
set(NCU_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/test/bench_sgemm.py" CACHE FILEPATH "")

if(NCU_EXECUTABLE)
    message(STATUS "Found Nsight Compute: ${NCU_EXECUTABLE}")

    # 设置报告的文件名和输出目录
    set(NCU_OUTPUT_NAME "ncu_SGEMM_v1" CACHE STRING "Base name of the ncu report")
    
    set(NCU_OUTPUT_DIR "${PROJECT_SOURCE_DIR}/reports/ncu_reports" CACHE PATH "Directory where ncu reports are saved")
    #创建输出目录
    file(MAKE_DIRECTORY ${NCU_OUTPUT_DIR})


    # ncu编译参数
    set(_ncu_args
        --set full
        --import-source on

        --target-processes all
        --nvtx --nvtx-include "profile/"
        
        -f
        -o "${NCU_OUTPUT_DIR}/${NCU_OUTPUT_NAME}"
        ${NCU_PYTHON} ${NCU_SCRIPT}
    )
    # nvcc编译参数
    target_compile_options(cudakernels_torch PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)


    # 遇到 ERR_NVGPUCTRPERM（性能计数器权限不足）时开启，用 sudo 运行 ncu
    option(NCU_USE_SUDO "Run ncu with sudo (needs passwordless sudo or an interactive terminal)" OFF)

    set(_ncu_cmd ${NCU_EXECUTABLE})

    if(NCU_USE_SUDO)
        list(INSERT _ncu_cmd 0 sudo)
    endif()



    # 添加自定义target
    add_custom_target(profile_ncu
        COMMAND ${_ncu_cmd} ${_ncu_args}
        DEPENDS cudakernels_torch
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Running Nsight Compute profiling -> ${NCU_OUTPUT_DIR}/${NCU_OUTPUT_NAME}.ncu-rep"
        VERBATIM
    )
    add_custom_command(TARGET profile_ncu POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Profile report saved to: ${NCU_OUTPUT_DIR}/${NCU_OUTPUT_NAME}.ncu-rep"
    )
else()
    message(WARNING "Nsight Compute not found in PATH. Profile target will not be available.")
endif()



set(NSYS_OUTPUT_DIR "${PROJECT_SOURCE_DIR}/reports/nsys_reports")

add_custom_target(profile_nsys
    COMMAND nsys profile
            --trace=cuda,nvtx,osrt,cuDNN
            --force-overwrite=true
            --output="${NSYS_OUTPUT_DIR}/nsys_SGEMM_v1"
            ${NCU_PYTHON} ${NCU_SCRIPT}
    DEPENDS cudakernels_torch
    COMMENT "Running nsys profile on my_target..."
)

add_custom_command(TARGET profile_nsys POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Profile report has saved."
)
