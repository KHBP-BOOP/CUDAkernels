# 将 `sgemm_block_tiling_v1` 注册为 PyTorch 算子（stable ABI）并全量分层重构

> 设计文档。**仓库当前尚未按本文档改动**；「实现步骤」是待执行的施工单。

## Context

**目标**：在 Python 里通过 `torch.ops.CUDAkernels.sgemm_v1(a, b)` 调用仓库里的 CUDA kernel `sgemm_block_tiling_v1`，与 `torch.matmul`(cuBLAS) 做精度/性能对比；同时把当前扁平的 `include/ + src/` 结构重组为分层项目。

**当前状态（关键问题）**：
1. `src/SGEMM.cu:697-707` 已开始写 stable ABI 注册，但**根本无法编译**：
   `m.def("sgemm_v1(Tensor A, Tensor B, Tensor C, int M, int N, int K) -> void")` 声明的是 `Tensor` 形参，而 `TORCH_BOX(&launch_sgemm_thread_tiling_v1)` 被应用到一个签名为 `(const float*, const float*, float*, int,int,int)` 的函数上。`TORCH_BOX` 通过 `infer_function_traits_t` 推导形参并做 `unbox_to_tuple<...>`，而 `stableivalue_conversions.h` 里**不存在** `StableIValue → const float*` 的转换 → 模板实例化即报错。
2. 扩展根本没有构建目标：`CMakeLists.txt` 只产出可执行文件 `EXEFILE`，链接行里没有任何 torch 库。
3. 整个 kernel 文件因为 `#include <torch/csrc/stable/library.h>` 而耦合了 torch —— 违背了 kernel 层应独立的初衷。
4. launcher 硬编码默认流（`<<<grid, block>>>` 无 stream 参数）。在 PyTorch 下算子必须跑在 `torch.cuda.current_stream()` 上，否则在 `torch.cuda.stream(...)` 上下文里会与其它算子产生竞态。**这是正确性问题，不是风格问题**。

**已验证的环境事实**（torch 2.13.0+cu130 / CUDA 13.2 / sm_89）：
- stable ABI 能力足够：`torch::stable::empty`/`new_empty`（`ops.h:615`/`:924`）、`Tensor::mutable_data_ptr()/const_data_ptr()/is_contiguous()/size()/scalar_type()/get_device_index()`、`accelerator::DeviceGuard`、`accelerator::getCurrentStream()` 且 `Stream::nativeHandle()` 在 `TORCH_FEATURE_VERSION >= 2_13_0` 可用（本机满足）→ 可拿到裸 `cudaStream_t`。
- `macros.h` 提供 `STD_TORCH_CHECK` / `STD_CUDA_KERNEL_LAUNCH_CHECK()`。
- `aoti_torch_library_def` 等符号定义在 `torch/lib/libtorch_cpu.so`（`nm -D` 已验证）→ **扩展不需要链接 libtorch**，符号在 `dlopen` 时从已加载的 torch 解析。

---

## 目标目录结构

```
CMakeLists.txt            # 三个 target
CMakePresets.json         # 不变
.gitignore                # 加 python 构建产物
README.md

kernels/                          → target: cuda_kernels (STATIC)
  include/cudakernels/
    sgemm.h                       # launcher 声明（仅 POD，无 torch）
    cuda_check.h                  # CUDA_CHECK / STD_CUDA_KERNEL_LAUNCH_CHECK 封装
    vec_add.h, tree_reduction.h, vector_dot_product.h    # 由 include/ 平移
  device/
    sgemm_block_tiling_v1.cuh     # __global__ 模板定义（header！见下方约束）
    sgemm_thread_tiling_v3/v4/v5.cuh
    load_tile.cuh                 # load_tile_A / load_tile_B / transposed 版
  host/
    sgemm_launch.cu               # include device/*.cuh，实例化模板并启动
    vec_add.cu, tree_reduction.cu, vector_dot_product.cu  # 平移（仍未编译）

bindings/                         → target: cudakernels_torch (SHARED)
  torch_binding.cu                # stable ABI 适配层 + STABLE_TORCH_LIBRARY 注册

apps/                             → target: EXEFILE
  sgemm/main.cc                   # 原 src/Main.cc
  sgemm/sgemm_test.cu             # 原 src/testSGEMM.cu（CPU double 参考 + 校验）

python/
  pyproject.toml                  # 纯 Python 包（不编译 C++，见下）
  cudakernels/__init__.py         # load_library + 导出 sgemm_v1

tests/
  test_sgemm.py                   # pytest：精度 + 计时

docs/                             # 不变
```

已存在的空目录 `binding/` 改名为 `bindings/`；`pyTest/test.py`（死代码，只剩两行 import）删除。

---

## 关键约束：device/host 分层会破坏当前的无 RDC 链接方式

`src/SGEMM.cu:627-630` 的注释是本次重构的硬约束：

> 直接跨翻译单元链接 `__global__` 模板实例化存在可见性问题（rdc=false 模式下模板实例化的 host stub 默认具有内部链接属性），因此测试代码通过本函数间接启动核函数

也就是说当下 `rdc=false`，`__global__` 模板的 host stub 是**内部链接**的 —— 谁实例化，谁才能启动。

因此 `kernels/device/` 与 `kernels/host/` 不能简单地拆成两个 `.cu`。处理方式：

**采用「模板放 header」**：`kernels/device/sgemm_block_tiling_v1.cuh` 里放完整的模板定义；`kernels/host/sgemm_launch.cu` `#include` 它并在**同一个 TU 内**实例化 + 启动。因为实例化与启动处于同一 TU，host stub 可见，无需 RDC；而不同 TU 各自实例化同名模板时，stub 是内部链接，也不会产生重复符号冲突。这样可以真正做到 device/host 目录分离，且**不需要打开 `CUDA_SEPARABLE_COMPILATION`**。

> **回退方案**（若实例化迁移后出现链接/启动问题）：保持 kernel 模板与其 launcher 在同一个 `.cu` 内（即今天的做法），文件放 `kernels/src/sgemm_v1.cu`，目录上只保留 `kernels/include` + `kernels/src` 两层。此方案零风险但分层不彻底。**先按 header 方案做，验证 `EXEFILE` 输出仍为 PASS；一旦异常立即回退。**

同时：**不要**开启 `CUDA_SEPARABLE_COMPILATION` / `CUDA_RESOLVE_DEVICE_SYMBOLS`。

---

## 实现步骤

### 1. 解耦并修复注册（先让仓库能编译）

- `src/SGEMM.cu` → `kernels/`：
  - 删除 `#include <torch/csrc/stable/library.h>`（第 2 行）与整段 `STABLE_TORCH_LIBRARY` / `STABLE_TORCH_LIBRARY_IMPL`（697-707 行）。**kernel 层不得再出现任何 torch 头文件。**
  - 把 `sgemm_block_tiling_v1` 模板（121-211 行）移入 `kernels/device/sgemm_block_tiling_v1.cuh`；`load_tile_*` 辅助函数移入 `kernels/device/load_tile.cuh`；其余 kernel 模板同样处理。
  - 把 launcher 移入 `kernels/host/sgemm_launch.cu`（`#include` 上面两个 `.cuh`）。
  - 保留文件级 `namespace CUDAkernels`（未提交的改动，一起提交）。

- **给 v1 增加带 stream 的重载**（不能只加默认参数：默认参数不改变函数类型，`&launch_sgemm_thread_tiling_v1` 会变成 7 参函数指针，无法赋给 `testSGEMM.cu:78` 的 `FUNC = void(*)(const float*,const float*,float*,int,int,int)`）：

```cpp
// kernels/include/cudakernels/sgemm.h
void launch_sgemm_thread_tiling_v1(const float *A, const float *B, float *C,
                                   int M, int N, int K);                       // 原有 6 参，FUNC 仍匹配
void launch_sgemm_thread_tiling_v1(const float *A, const float *B, float *C,
                                   int M, int N, int K, cudaStream_t stream);  // 新增，供 torch 适配层用
```

```cpp
// kernels/host/sgemm_launch.cu
void launch_sgemm_thread_tiling_v1(const float *A, const float *B, float *C, int M, int N, int K) {
    launch_sgemm_thread_tiling_v1(A, B, C, M, N, K, static_cast<cudaStream_t>(nullptr));
}

void launch_sgemm_thread_tiling_v1(const float *A, const float *B, float *C, int M, int N, int K,
                                   cudaStream_t stream) {
    constexpr int BM = 64, BN = 64, BK = 4, BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_block_tiling_v1<BM, BN, BK, BLOCK_SIZE>
        <<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
```

- `apps/sgemm/sgemm_test.cu` 里 `CUDAkernels::launch_sgemm_thread_tiling_v1` 会按目标类型自动选中 6 参重载，**无需改动**。

### 2. CMake 拆分

保持 1-21 行（project / C++20 / CUDA20 / `CMAKE_CUDA_ARCHITECTURES 89` / release flags）与 50-109 行的 `profile_ncu` / `profile_nsys` 块（`DEPENDS EXEFILE` 继续有效）。替换 26-46 行：

```cmake
# ---- 1) kernel 层：设备代码 + 主机启动封装，不含 torch ----
add_library(cuda_kernels STATIC
    kernels/host/sgemm_launch.cu
    # kernels/host/vec_add.cu ...
)
set_target_properties(cuda_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)  # 要链进 .so
target_include_directories(cuda_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/kernels/include)
target_compile_options(cuda_kernels PRIVATE -lineinfo)   # -lineinfo 随 SGEMM.cu 一起搬，否则 ncu 丢源码关联

# ---- 2) C++ 测试程序 ----
add_executable(EXEFILE apps/sgemm/main.cc apps/sgemm/sgemm_test.cu)
target_link_libraries(EXEFILE PRIVATE cuda_kernels)
target_compile_options(EXEFILE PRIVATE -lineinfo)

# ---- 3) torch stable-ABI 扩展 ----
option(CUDAKERNELS_BUILD_TORCH_EXT "Build the PyTorch stable-ABI extension" ON)
if(CUDAKERNELS_BUILD_TORCH_EXT)
    # 不再硬编码 pyTest/.venv/lib64/...；可用 -DPython3_EXECUTABLE= 覆盖
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
                "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'include'))"
        RESULT_VARIABLE _rc OUTPUT_VARIABLE TORCH_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _rc EQUAL 0 OR NOT EXISTS "${TORCH_INCLUDE_DIR}/torch/csrc/stable/library.h")
        message(FATAL_ERROR "需要 torch>=2.13 (stable ABI)。得到 '${TORCH_INCLUDE_DIR}'")
    endif()
    message(STATUS "Torch stable-ABI include: ${TORCH_INCLUDE_DIR}")

    add_library(cudakernels_torch SHARED bindings/torch_binding.cu)   # 注册 TU 必须是 .so 的直接源文件
    set_target_properties(cudakernels_torch PROPERTIES
        OUTPUT_NAME cudakernels_torch
        POSITION_INDEPENDENT_CODE ON
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON      # stable 头文件面向 C++17，降低风险
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    target_include_directories(cudakernels_torch PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/include "${TORCH_INCLUDE_DIR}")
    target_link_libraries(cudakernels_torch PRIVATE cuda_kernels)
    # 刻意不 find_package(Torch) / 不链接 libtorch：stable ABI 的符号在 dlopen 时解析
endif()
```

要点：
- **注册必须直接编译进 `.so`**。`STABLE_TORCH_LIBRARY` 展开出静态初始化对象（`library.h:355-384`）；若其所在目标文件被归档进 `libcuda_kernels.a` 且无人引用，链接器不会抽取它，注册就静默失效。所以 `bindings/torch_binding.cu` 必须列在 `add_library(... SHARED ...)` 的直接源里。适配层引用了 `launch_sgemm_thread_tiling_v1`，会把 `cuda_kernels` 里的目标文件（连带 fatbin）拉进来。
- kernel 库用 **STATIC** 即可：`rdc=false` 下每个 `.cu` 自带 fatbin 与 host stub，归档成员被抽取时其 fatbin 一并进入消费者（exe 或 `.so`），`EXEFILE` 与扩展各持一份（几百 KB），无 RDC、无设备链接。
- 不需要 `-Wl,--allow-shlib-undefined`（GNU ld 对 `.so` 默认允许未定义符号）；只要别加 `-Wl,--no-undefined`。

### 3. 适配层 + 注册（`bindings/torch_binding.cu`）

```cpp
// 本 TU 必须直接编译进扩展 .so，否则 STABLE_TORCH_LIBRARY 静态初始化会被丢弃
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>
#include <cuda_runtime_api.h>
#include <array>
#include <cstdint>
#include <limits>
#include "cudakernels/sgemm.h"

namespace cudakernels_torch {
using torch::stable::ScalarType;
using torch::stable::Tensor;

namespace {
void check_f32_cuda_2d_contig(const Tensor& t, const char* name) {
    STD_TORCH_CHECK(t.defined(),               name, " 必须是已定义张量");
    STD_TORCH_CHECK(t.dim() == 2,              name, " 必须是 2 维");
    STD_TORCH_CHECK(t.scalar_type() == ScalarType::Float, name, " 必须是 float32");
    STD_TORCH_CHECK(t.is_contiguous(),         name, " 必须是连续内存");
    STD_TORCH_CHECK(t.is_cuda(),               name, " 必须在 CUDA 设备上");
}
}  // namespace

// 必须外部链接：TORCH_BOX 把它的地址作为非类型模板实参
Tensor sgemm_v1(const Tensor& a, const Tensor& b) {
    check_f32_cuda_2d_contig(a, "a");
    check_f32_cuda_2d_contig(b, "b");
    STD_TORCH_CHECK(a.get_device_index() == b.get_device_index(), "a 与 b 必须在同一设备");

    const int64_t M = a.size(0), K = a.size(1);
    STD_TORCH_CHECK(b.size(0) == K, "形状不匹配: b 的第 0 维应等于 ", K);
    const int64_t N = b.size(1);
    constexpr int64_t kMax = std::numeric_limits<int32_t>::max();
    STD_TORCH_CHECK(M > 0 && N > 0 && K > 0 && M <= kMax && N <= kMax && K <= kMax,
                    "M/N/K 必须为正且不超过 int32 范围");

    const auto dev = a.get_device_index();
    torch::stable::accelerator::DeviceGuard guard(dev);          // 绑定当前设备

    std::array<int64_t, 2> sizes{M, N};
    Tensor c = torch::stable::new_empty(a, sizes, ScalarType::Float);   // 继承 a 的 device/layout

    auto stream = torch::stable::accelerator::getCurrentStream(dev);    // 必须用 torch 当前流
    cudaStream_t raw = reinterpret_cast<cudaStream_t>(stream.nativeHandle());

    CUDAkernels::launch_sgemm_thread_tiling_v1(
        a.const_data_ptr<float>(), b.const_data_ptr<float>(), c.mutable_data_ptr<float>(),
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), raw);

    STD_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}
}  // namespace cudakernels_torch

STABLE_TORCH_LIBRARY(CUDAkernels, m) {
    m.def("sgemm_v1(Tensor a, Tensor b) -> Tensor");
}
STABLE_TORCH_LIBRARY_IMPL(CUDAkernels, CUDA, m) {
    m.impl("sgemm_v1", TORCH_BOX(&cudakernels_torch::sgemm_v1));
}
```

- `const_data_ptr<float>()` / `mutable_data_ptr<float>()` 需 `TORCH_FEATURE_VERSION >= 2_10_0`；`nativeHandle()` 需 `>= 2_13_0` —— 本机 2.13.0 满足，**不要**定义低于 2.13 的 `TORCH_TARGET_VERSION`。
- 必须在 `CUDAkernels, CUDA` 下 `impl`（不是 `CompositeExplicitAutograd`），因为 kernel 只在 CUDA 上有意义。
- 单返回值 → schema 必须写 `-> Tensor`（boxer 断言 `num_outputs==1`）；若将来加 `void` 返回的 `sgemm_v1_out`，schema 必须写 `-> ()`。
- 落地时按 `ops.h:615`(empty) / `:924`(new_empty) 的**实际重载签名**核一遍调用形式。

### 4. Python 侧

`python/cudakernels/__init__.py`：import 时定位并加载已构建的 `.so`，导出算子句柄。

```python
import os, torch
from pathlib import Path

def _locate_ext() -> Path:
    if (env := os.environ.get("CUDAKERNELS_EXT")):
        return Path(env)
    root = Path(__file__).resolve().parents[2]
    hits = sorted((root / "out" / "build").glob("*/libcudakernels_torch.so"))
    if not hits:
        raise RuntimeError("未找到 libcudakernels_torch.so，请先 cmake --build")
    return hits[-1]

torch.ops.load_library(str(_locate_ext()))
sgemm_v1 = torch.ops.CUDAKernels.sgemm_v1
```

`python/pyproject.toml`：纯 Python 包（CMake 负责编译 `.so`，pip 只装 wrapper）：

```toml
[project]
name = "cudakernels"
version = "0.1.0"
dependencies = ["torch>=2.13"]
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
packages = ["cudakernels"]
```

> 用 `uv pip install -e python/` 安装。若日后想让 `pip install` 顺带编译 C++，再换 `scikit-build-core` 后端——本次不做。

`tests/test_sgemm.py`（pytest，需先 `uv pip install pytest`）：对 `(1024,1024,1024)`、非 tile 整数倍 `(1000,1004,1008)`、`(129,257,513)`、退化 `(1,4,4)` / `(63,65,3)` 逐一对 `torch.matmul` 比对，容差沿用仓库 `verify_result` 的 `atol=rtol=1e-2`，并先关掉 TF32（`torch.backends.cuda.matmul.allow_tf32 = False`）使对比有意义；再加 CPU 张量 / float64 的负例断言抛 `RuntimeError`；最后用 `torch.cuda.Event` 对 `sgemm_v1` 与 `a @ b` 计时出 TFLOP/s。

> 说明：cuBLAS fp32 的累加顺序与我们的 kernel 不同，结果是 ~1e-5..1e-4 相对差，**不可能逐位相等**；别把 `allclose` 写成 `equal`。

### 5. 收尾

- `.gitignore` 增加 `python/**/__pycache__/`、`*.egg-info/`；`out/**`、`reports/**` 已覆盖 `.so`。
- `README.md`（当前 0 字节）补构建/加载/测试三条命令。
- 删除空的 `pyTest/`（`.venv` 的处理见下）。
- 把未提交的 `namespace CUDAkernels` 改动与本次拆分一起提交。

---

## 验证

```bash
cd /home/khbp/codes/CUDAkernels
cmake --preset GCC-13.3.0-x86_64-linux-gnu          # 应打印 Torch stable-ABI include
cmake --build out/build/GCC-13.3.0-x86_64-linux-gnu -j

# 1) C++ 通路未被破坏：应仍为 PASS
./out/build/GCC-13.3.0-x86_64-linux-gnu/EXEFILE

# 2) ncu 通路未被破坏（确认 -lineinfo 跟着 kernel 库走了）
cmake --build out/build/GCC-13.3.0-x86_64-linux-gnu --target profile_ncu

# 3) 算子已注册
PYTHONPATH=python .venv/bin/python -c \
  "import cudakernels, torch; print(torch.ops.CUDAkernels.sgemm_v1.default._schema)"

# 4) 精度 + 计时
.venv/bin/python -m pytest tests/ -v
```

判定标准：`EXEFILE` 与拆分前输出一致（PASS）；`_schema` 打印 `sgemm_v1(Tensor a, Tensor b) -> Tensor`；pytest 全绿且非 2 的幂形状也过。

---

## 风险与注意事项

1. **`TORCH_BOX` 形参约束（正在踩的坑）**：被 box 的函数只能收 stable-ABI 类型（`Tensor`/`int64_t`/`bool`/`std::optional<...>`/`string_view`/`HeaderOnlyArrayRef`）。永远 box 适配函数，绝不 box 裸指针 launcher。
2. **device/host 拆分与 RDC**：见上文「关键约束」。模板必须进 header、在与 launcher 同一 TU 内实例化。任何跨 TU 的「只声明后启动」都会以内部链接 stub 失败。若出现异常，回退到 kernel+launcher 同 `.cu`。
3. **注册可见性**：注册 TU 必须直接编译进 `.so`。放进静态库需 `$<LINK_LIBRARY:WHOLE_ARCHIVE,cuda_kernels>` 或 `-Wl,--whole-archive` 才不会被丢弃。
4. **加载顺序**：必须先 `import torch`，再 `torch.ops.load_library(<绝对路径>)`。用裸 `ctypes.CDLL` 且不带 `RTLD_GLOBAL` 会解析不到 `aoti_torch_*`。
5. **torch cu130 vs 本地 CUDA 13.2**：扩展的 `NEEDED libcudart.so.13` 在运行时会绑到 torch 已加载的 13.0.96；我们把 13.2 编出来的 cubin 跑在 13.0 runtime 上。CUDA 13.x 次版本兼容 + 只用长期稳定 API，正常应无碍，但这是**最可能出意外的一点**。缓解：`set(CMAKE_CUDA_ARCHITECTURES "89-real;89-virtual")` 额外嵌 PTX 作 JIT 兜底；真出问题就把 venv 的 `nvidia/cu13/include` 加进 binding TU 的 include 并对齐 runtime 版本。
6. **`nativeHandle()` 的版本门槛**：仅 `TORCH_FEATURE_VERSION >= 2_13_0` 存在；低于此版本 stable ABI 没有受支持的途径拿到裸 `cudaStream_t`。故构建期应要求 torch ≥ 2.13（当前 2.13.0）。
7. **`.venv` 位置**：现有虚拟环境在 `pyTest/.venv`（已被 gitignore）。移动到根 `.venv/` 会更整洁，但 uv 的 venv 内嵌绝对路径，移动需重建（要重新拉取数 GB 的 torch）。**建议**：保持原地不动，CMake 用 `find_package(Python3)` + `-DPython3_EXECUTABLE=` 覆盖即可两边兼容；若愿意重建再迁到根 `.venv/`。上面命令里的 `.venv/bin/python` 请按实际路径替换。
8. **v3/v4/v5 的 grid 不一致 bug**：`src/SGEMM.cu:649-694` 用局部 `BM=BN=64,BK=4` 算 grid/block，却实例化 `<128,128,8,256,8,4,8,8>`，grid 只覆盖 C 的左上角。本次只绑定 v1；迁移时**原样搬运**、不要顺手「修好」，另开一次提交处理。
9. **`sgemm_test.cu` 头注释与实现不符**：注释声称有 cudaEvent 计时与 TFLOPS，实际没有任何计时代码（`PerfAnaly` 只是多启一次）。真实计时本次落在 Python 侧，别指望 C++ 程序给出 TFLOPS。
10. **`-lineinfo` 必须跟着 kernel 走**：`SGEMM.cu` 进静态库后，`-lineinfo` 要加到 `cuda_kernels`，否则 ncu 丢失源码关联（`profile_ncu` target 本身不用改）。
11. **张量校验**：非连续输入当前直接报错（推荐，显式优于隐式）；若要宽容，改用 `torch::stable::contiguous(a)` 取连续副本。
