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
    C_custom = cudakernels.sgemm_v1(A, B)
    C_torch = torch.mm(A, B)

    # 默认值 rtol=1.3e-6 atol=1e-5
    torch.testing.assert_close(C_custom, C_torch, rtol = 1e-4, atol = 1e-4)

def various_input_correctness_test(M: int, K: int, N: int) -> None:

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

def input_random_performance_analysis(M: int, K: int, N: int) -> None :

    A = torch.randn((M, K), device="cuda", dtype=torch.float32)
    B = torch.randn((K, N), device="cuda", dtype=torch.float32)


    # warm up
    for _ in range(3):
        C_custom = cudakernels.sgemm_v1(A, B)
        C_torch_mm = torch.mm(A, B)


    # analysis
    torch.cuda.nvtx.range_push("profile")

    C_custom = cudakernels.sgemm_v1(A, B)
    C_torch_mm = torch.mm(A, B)

    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()




# a）精度测试
various_input_correctness_test(2, 2, 2)
various_input_correctness_test(7, 13, 5)
various_input_correctness_test(31, 63, 47)
various_input_correctness_test(64, 64, 64)
various_input_correctness_test(256, 512, 128)
various_input_correctness_test(127, 129, 65)
various_input_correctness_test(1024, 1024, 1024)


# b）Nsight Compute 性能分析
input_random_performance_analysis(1024, 1024, 1024)
