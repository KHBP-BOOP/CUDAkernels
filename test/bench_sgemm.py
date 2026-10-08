"""sgemm_block_tiling_v1 vs cuBLAS baseline。

用法：
    test/.venv/bin/python test/bench_sgemm.py          # 只做精度校验
    cmake --build out/build/<preset> --target profile_ncu   # 出 ncu-rep
"""
import sys
from pathlib import Path

import torch

# 不依赖 PYTHONPATH：sudo 下环境变量会被清掉，导入路径必须写死在脚本里
try:
    import cudakernels
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    import cudakernels

# 真 FP32，否则 cuBLAS 在 sm_89 上会走 TF32 tensor core，与我们的 kernel 不可比
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

M = N = K = 1024
torch.manual_seed(0)
a = torch.randn(M, K, device="cuda")
b = torch.randn(K, N, device="cuda")

# 预热：让 cuBLAS 完成启发式选核，也完成我们 kernel 的首次加载。
# 必须在 NVTX 窗口之外，否则这些 launch 也会被 ncu 抓进去。
for _ in range(5):
    ref = a @ b
    out = cudakernels.sgemm_v1(a, b)
torch.cuda.synchronize()

# 精度校验也在窗口外
max_err = (out - ref).abs().max().item()
print(f"max|Δ| = {max_err:.3e}")
assert torch.allclose(out, ref, atol=1e-2, rtol=1e-2), "精度不达标"

# ---- ncu 的 profile 窗口：窗口内恰好两次 kernel launch，无其它 ----
torch.cuda.nvtx.range_push("profile")
ref = a @ b                        # cuBLAS baseline
out = cudakernels.sgemm_v1(a, b)   # sgemm_block_tiling_v1<64,64,4,256>
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
