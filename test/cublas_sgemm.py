""" 约定：

计算中小尺寸矩阵时warmup 3次，不锁定GPU时钟频率
计算中大尺寸warmup 1次，不锁频
计算大尺寸不设warmup，锁频(否则RSD难以低于2)

> 大量warmup导致GPU升频并稳定后，会由于功耗、温度等因素引入额外噪音



初步结论：

在较大尺寸（M/N/K均大于1024左右时）矩阵表现极佳，尺寸良好的情况下，可稳定超越cuBLAS的绝对执行时间

在较小尺寸（M/N/K任一小于128或256左右时）矩阵表现欠佳，但部分尺寸下仍达90%左右的性能

"""

import nsight
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



@nsight.analyze.kernel(runs=10,
                       clock_control="base",
                       output_csv=True,
                       output_prefix="/home/khbp/codes/CUDAkernels/test/csv/out/p2_benmk4_",
                       cache_control="all", # default
                       metrics=["gpu__time_duration.sum"] # default
                        )
def warmup_benchmark(M: int, K: int, N: int) -> None :

    A = torch.randn((M, K), device="cuda", dtype=torch.float32)
    B = torch.randn((K, N), device="cuda", dtype=torch.float32)


    # CUDAkernels warm up
    for _ in range(0):
        _ = cudakernels.sm89_fp32_gemm(A, B)

    # benchmark
    with nsight.annotate(f"sm89_fp32_gemm_{M}_{K}_{N}"):
        C_custom = cudakernels.sm89_fp32_gemm(A, B)


    # cuBLAS warm up
    for _ in range(0):
        _ = torch.mm(A, B)

    # benchmark
    with nsight.annotate(f"cuBLAS_{M}_{K}_{N}"):
        C_torch_mm = torch.mm(A, B)


if __name__ == "__main__":

    # benchmark & print
    # 可被整除
    # warmup_benchmark(256, 256, 256)
    # warmup_benchmark(512, 512, 512)
    # warmup_benchmark(256, 1024, 1024)

    # warmup_benchmark(1024, 1024, 1024)
    # warmup_benchmark(2048, 2048, 2048)
    # warmup_benchmark(3072, 3072, 3072)
    # warmup_benchmark(4096, 4096, 4096)
    # warmup_benchmark(6144, 6144, 6144)
    # warmup_benchmark(8192, 8192, 8192)


    # warmup_benchmark(256, 10240, 2048)
    warmup_benchmark(512, 8192, 3072)
    warmup_benchmark(512, 10240, 1024)

    # warmup_benchmark(256, 2048, 3072)
    # warmup_benchmark(10240, 1024, 256)
    # warmup_benchmark(10240, 256, 256)
    # warmup_benchmark(1024, 1024, 2048)
    # warmup_benchmark(1024, 256, 3072)
    # warmup_benchmark(256, 2048, 10240)
    warmup_benchmark(512, 3072, 8192)
    
    



    # 小M
    # warmup_benchmark(32, 4096, 4096)

    # 小K
    # warmup_benchmark(4096, 32, 4096)

    # 小N
    # warmup_benchmark(4096, 4096, 32)


    # # 大M
    # warmup_benchmark(4096, 32, 32)
    
    # # 大K
    # # warmup_benchmark(64, 4096, 16)

    # # 大N
    # warmup_benchmark(32, 32, 4096)



    # # decode
    # # (64, 2048, 7168) (128, 24576, 1536)
    # warmup_benchmark(64, 2048, 7168)
    # warmup_benchmark(128, 24576, 1536)

    # # small prefill 与 speculative decode
    # # (32, 12288, 4096) (16, 14336, 4096)
    # warmup_benchmark(32, 12288, 4096)
    # warmup_benchmark(16, 14336, 4096)

    # # larger prefill
    # # (4096, 2048, 7168) (4096, 7168, 16384)
    # warmup_benchmark(4096, 2048, 7186)
    # warmup_benchmark(4096, 7168, 16384)

