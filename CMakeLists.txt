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

set(CMAKE_BUILD_TYPE debug)
set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG" CACHE STRING "" FORCE)




#生成可执行文件
add_executable(EXEFILE)

target_sources(EXEFILE
PRIVATE
    src/Main.cc
    # src/vecAdd.cu
    # src/vectorDotProduct.cu
    # src/DEVICE.cu
    # src/treeReduction.cu
    src/SGEMM.cu
    src/testSGEMM.cu
)


target_include_directories(EXEFILE
PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)



# 优先使用 CUDAToolkit 自带的 ncu（即使 CUDA bin 不在 PATH 中也能找到）
find_program(NCU_EXECUTABLE ncu HINTS "${CUDAToolkit_BIN_DIR}")
if(NCU_EXECUTABLE)
    message(STATUS "Found Nsight Compute: ${NCU_EXECUTABLE}")

    # 报告基础文件名（ncu 会自动追加 .ncu-rep 后缀）和输出目录
    set(NCU_OUTPUT_NAME "ncu_SGEMM_v1_debug" CACHE STRING "Base name of the ncu report")
    set(NCU_OUTPUT_DIR "${PROJECT_SOURCE_DIR}/reports/ncu_reports" CACHE PATH "Directory where ncu reports are saved")

    # # 可选：只对匹配的 kernel 做 profile（regex 语法，如 regex:blockReduceSum）；留空则 profile 全部 kernel
    # set(NCU_KERNEL_FILTER "" CACHE STRING "Optional ncu --kernel-name regex; empty profiles all kernels")
    # # 可选：透传给 ncu 的附加参数，例如 "--launch-count 1 --launch-skip 0"
    # set(NCU_EXTRA_ARGS "" CACHE STRING "Extra arguments passed to ncu")
    # 遇到 ERR_NVGPUCTRPERM（性能计数器权限不足）时开启，用 sudo 运行 ncu
    option(NCU_USE_SUDO "Run ncu with sudo (needs passwordless sudo or an interactive terminal)" OFF)

    # 确保输出目录存在
    file(MAKE_DIRECTORY ${NCU_OUTPUT_DIR})

    set(_ncu_cmd ${NCU_EXECUTABLE})
    if(NCU_USE_SUDO)
        list(INSERT _ncu_cmd 0 sudo)
    endif()



    set(_ncu_args --set full --import-source on -f -o "${NCU_OUTPUT_DIR}/${NCU_OUTPUT_NAME}" $<TARGET_FILE:EXEFILE>)

    target_compile_options(EXEFILE PRIVATE -g $<$<COMPILE_LANGUAGE:CUDA>:-G>)



    # 3. 添加自定义分析 target
    add_custom_target(profile_ncu
        COMMAND ${_ncu_cmd} ${_ncu_args}
        DEPENDS EXEFILE
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Running Nsight Compute profiling on EXEFILE -> ${NCU_OUTPUT_DIR}/${NCU_OUTPUT_NAME}.ncu-rep"
        VERBATIM
    )
    add_custom_command(TARGET profile_ncu POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Profile report saved to: ${NCU_OUTPUT_DIR}/${NCU_OUTPUT_NAME}.ncu-rep"
    )
else()
    message(WARNING "Nsight Compute (ncu) not found in PATH. Profile target will not be available.")
endif()



set(NSYS_OUTPUT_DIR "${PROJECT_SOURCE_DIR}/reports/nsys_reports")

add_custom_target(profile_nsys
    COMMAND nsys profile
            --trace=cuda,nvtx,osrt,cuDNN
            --force-overwrite=true
            --output="${NSYS_OUTPUT_DIR}/nsys_SGEMM_v1"
            $<TARGET_FILE:EXEFILE>
    DEPENDS EXEFILE
    COMMENT "Running nsys profile on my_target..."
)

add_custom_command(TARGET profile_nsys POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Profile report has saved."
)
