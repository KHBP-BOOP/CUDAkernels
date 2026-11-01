import os
from pathlib import Path

import torch  # 必须先 import torch，libtorch_*.so 就位后 dlopen 才能解析 aoti_torch_* 符号


def _locate_ext() -> Path:
    """ 定位 CMake 产出的扩展 .so CUDAKERNELS_EXT 可覆盖 """
    if env := os.environ.get("CUDAKERNELS_EXT"):
        return Path(env)
    
    root = Path(__file__).resolve().parents[2]           # python/cudakernels/__init__.py -> 仓库根
    hits = sorted((root / "out" / "build").glob("*/libcudakernels_torch.so"))

    if not hits:
        raise RuntimeError("未找到 libcudakernels_torch.so，请先 cmake --build")
    
    return hits[-1]

# 将kernel与相关wrapper生成为一个动态库
# 操作系统调用dlopen()，将该共享库加载至import该python包的线程
# 此时库中所有静态成员的动态初始化会被触发，通过c++ API将算子的schema和实现写进pytorch的Dispatcher
torch.ops.load_library(str(_locate_ext()))

sm89_fp32_gemm = torch.ops.CUDAkernels.sm89_fp32_gemm
__all__ = ["sm89_fp32_gemm"]
