// torch stable-ABI 的适配层 + 注册层

#include <torch/csrc/stable/accelerator.h>   // DeviceGuard / getCurrentStream / nativeHandle
#include <torch/csrc/stable/library.h>       // STABLE_TORCH_LIBRARY / TORCH_BOX
#include <torch/csrc/stable/ops.h>           // new_empty
#include <torch/csrc/stable/tensor.h>        // torch::stable::Tensor
#include <torch/headeronly/util/Exception.h> // STD_TORCH_CHECK

#include <cuda_runtime_api.h>                // cudaStream_t

#include <array>
#include <cstdint>
#include <limits>

namespace CUDAkernels {

void instantiated_sgemm_block_tiling_v1(cudaStream_t s, const float *A, const float *B, float *C, int M, int N, int K);

} // namespace CUDAkernels




namespace cudakernels_torch {

using torch::stable::ScalarType;
using torch::stable::Tensor;

namespace {
void check_f32_cuda_2d_contig(const Tensor& t, const char* name) {
    STD_TORCH_CHECK(t.defined(),                          name, " 必须是已定义张量");
    STD_TORCH_CHECK(t.dim() == 2,                         name, " 必须是 2 维张量");
    STD_TORCH_CHECK(t.scalar_type() == ScalarType::Float, name, " 必须是 float32");
    STD_TORCH_CHECK(t.is_contiguous(),                    name, " 必须是连续内存");
    STD_TORCH_CHECK(t.is_cuda(),                          name, " 必须在 CUDA 设备上");
}
}  // namespace

// 必须在匿名 namespace 之外：TORCH_BOX 把它的地址用作非类型模板实参。
Tensor sgemm_v1(const Tensor& a, const Tensor& b) {

    check_f32_cuda_2d_contig(a, "a");
    check_f32_cuda_2d_contig(b, "b");

    STD_TORCH_CHECK(a.get_device_index() == b.get_device_index(), "a 与 b 必须在同一设备");


    const int64_t M = a.size(0), K = a.size(1);
    STD_TORCH_CHECK(b.size(0) == K, "形状不匹配: b 的第 0 维应等于 ", K);
    const int64_t N = b.size(1);

    // launcher 与 kernel 内部都用 int 索引，这里收紧到 int32
    constexpr int64_t kMax = std::numeric_limits<int32_t>::max();
    STD_TORCH_CHECK(M >= 0 && N >= 0 && K >= 0 && M <= kMax && N <= kMax && K <= kMax,
                    "M/N/K 必须为正且不超过 int32 范围");


    const torch::stable::DeviceIndex dev = a.get_device_index();
    torch::stable::accelerator::DeviceGuard guard(dev);   // 绑定当前设备

    std::array<int64_t, 2> sizes{M, N};
    Tensor c = torch::stable::new_empty(a, sizes, ScalarType::Float);  // 继承 a 的 device

    // 必须走 torch 当前流，否则在 torch.cuda.stream(...) 下与其它算子竞态
    auto stream = torch::stable::accelerator::getCurrentStream(dev);
    cudaStream_t raw = reinterpret_cast<cudaStream_t>(stream.nativeHandle());

    CUDAkernels::instantiated_sgemm_block_tiling_v1(
        raw,
        a.const_data_ptr<float>(),
        b.const_data_ptr<float>(),
        c.mutable_data_ptr<float>(),
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));

    STD_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}

}  // namespace cudakernels_torch

// 声明算子的名称、形参列表、返回类型
STABLE_TORCH_LIBRARY(CUDAkernels, m) {
    m.def("sgemm_v1(Tensor a, Tensor b) -> Tensor");
}
// 挂到 CUDA dispatch key 下
STABLE_TORCH_LIBRARY_IMPL(CUDAkernels, CUDA, m) {
    m.impl("sgemm_v1", TORCH_BOX(&cudakernels_torch::sgemm_v1));
}
