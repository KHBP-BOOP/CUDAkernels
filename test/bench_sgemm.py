""" benchmark流程

1. import标准库、torch库、自定义的python库
2. 指定GEMM算子执行的精度
3. 精度测试 + 性能分析

"""

import sys
from pathlib import Path
import torch

try:
    import cudakernels
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    import cudakernels


assert torch.cuda.is_available()

torch.backends.cuda.matmul.fp32_precision="ieee"




def calculate_test(A: torch.Tensor, B: torch.Tensor):
    C_custom = cudakernels.sm89_fp32_gemm(A, B)
    C_torch = torch.mm(A, B)

    # 默认值 rtol=1.3e-6 atol=1e-5
    torch.testing.assert_close(C_custom, C_torch, rtol = 1e-4, atol = 1e-4)

def various_input_correctness_test(M: int, N: int, K: int) -> None:

    # 输入矩阵的元素值随机
    A = torch.randn((M, K), device="cuda", dtype=torch.float32)
    B = torch.randn((K, N), device="cuda", dtype=torch.float32)

    calculate_test(A, B)


    # 输入矩阵的元素值全为零
    A.zero_()
    B.zero_()

    calculate_test(A, B)


    # 输入矩阵的主对角线元素全为零
    # 部分测试用例中张量的形状对应为方阵，故次小节包含了对单位矩阵的测试
    A.fill_diagonal_(1.0)
    B.fill_diagonal_(1.0)

    calculate_test(A, B)

    print(f"(M, N, K) : {M}, {N}, {K} case passed!!")

def input_random_ncu_profiling(M: int, N: int, K: int) -> None :

    A = torch.randn((M, K), device="cuda", dtype=torch.float32)
    B = torch.randn((K, N), device="cuda", dtype=torch.float32)



    # CUDAkernels warm up
    for _ in range(3):
        C_custom = cudakernels.sm89_fp32_gemm(A, B)

    # profiling
    torch.cuda.nvtx.range_push("ncu_profile")
    C_custom = cudakernels.sm89_fp32_gemm(A, B)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()



    # cuBLAS warm up
    for _ in range(3):
        C_custom = cudakernels.sm89_fp32_gemm(A, B)

    # profiling
    torch.cuda.nvtx.range_push("ncu_profile")
    C_torch_mm = torch.mm(A, B)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()







# a）精度测试
# full-tile（多 stage、环形缓冲回绕）
various_input_correctness_test(64, 64, 64)
various_input_correctness_test(64, 64, 48)
various_input_correctness_test(64, 64, 80)
various_input_correctness_test(128, 128, 128)
various_input_correctness_test(256, 256, 256)
various_input_correctness_test(1024, 1024, 1024)

# partial-tile（M/N/K 边界）
various_input_correctness_test(48, 48, 48)
various_input_correctness_test(80, 80, 80)
various_input_correctness_test(63, 63, 63)
various_input_correctness_test(65, 65, 65)
various_input_correctness_test(127, 129, 65)
various_input_correctness_test(129, 127, 17)

# K 非 4 倍数、非方阵和极小尺寸
various_input_correctness_test(64, 68, 17)
various_input_correctness_test(65, 67, 15)
various_input_correctness_test(32, 96, 17)
various_input_correctness_test(2, 3, 2)
various_input_correctness_test(7, 5, 13)




# b）Nsight Compute 性能分析
# 2048 3072 4096 5120 6144
# 95%  95%  96%  96%
# 512 1024
# 75% 93%

input_random_ncu_profiling(16, 16, 16)
input_random_ncu_profiling(32, 32, 32)
input_random_ncu_profiling(64, 64, 64)
input_random_ncu_profiling(128, 128, 128)
input_random_ncu_profiling(256, 256, 256)
input_random_ncu_profiling(512, 512, 512)
input_random_ncu_profiling(1024, 1024, 1024)

input_random_ncu_profiling(2048, 2048, 2048)
input_random_ncu_profiling(3072, 3072, 3072)
input_random_ncu_profiling(4096, 4096, 4096)
input_random_ncu_profiling(5120, 5120, 5120)
input_random_ncu_profiling(6144, 6144, 6144)
input_random_ncu_profiling(8192, 8192, 8192)
