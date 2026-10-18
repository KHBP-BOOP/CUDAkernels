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

# 将共享库加载至当前线程，触发其中所有静态成员的静态初始化
# 此时执行静态初始化器，通过c++ API将算子的schema和实现注册进pytorch的Dispatcher
torch.ops.load_library(str(_locate_ext()))

sgemm_v1 = torch.ops.CUDAkernels.sgemm_v1
__all__ = ["sgemm_v1"]
