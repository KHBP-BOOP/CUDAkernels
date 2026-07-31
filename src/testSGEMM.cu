// testSGEMM.cu
// 主机端 SGEMM 测试：正确性校验 + 性能基准
//
// 核函数约束（来自 load_tile 的 LDG.128 与写回阶段的 STG.128）：
//   - K 必须为 4 的倍数（A、B 读取的 16B 对齐）
//   - N 必须为 4 的倍数（C 写回的 16B 对齐）
//   - M 任意（越界行由写回逻辑跳过）
//
// 测试内容：
//   1. 正确性：多组尺寸（含 M/N 非 128 倍数、K 非 8 倍数等边界）
//      与 CPU 双精度参考结果对比
//   2. 性能：4096×4096×4096 基准，输出平均耗时与 GFLOPS

#include "SGEMM.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(err), __FILE__, __LINE__);              \
            exit(-1);                                                          \
        }                                                                      \
    } while (0)

// 核函数模板参数（必须与 include/SGEMM.cuh 末尾 extern template 的实例化一致）
constexpr int BM = 128, BN = 128, BK = 8;
constexpr int BLOCK_SIZE = 256, Wx = 8, Wy = 4, TM = 8, TN = 8;

// CPU 双精度参考实现: C = A * B（行主序）
static void sgemm_ref_cpu(const std::vector<float> &A, const std::vector<float> &B,
                          std::vector<float> &C, int M, int N, int K) {
    std::fill(C.begin(), C.end(), 0.0f);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k) {
                sum += static_cast<double>(A[i * K + k]) * B[k * N + j];
            }
            C[i * N + j] = static_cast<float>(sum);
        }
    }
}

// 逐元素校验：绝对误差 ≤ 1e-5 或相对误差 ≤ 1e-4 视为一致
// 返回是否通过，并通过引用输出最大绝对/相对误差
static bool verify(const std::vector<float> &C_ref, const std::vector<float> &C_gpu,
                   int M, int N, double &max_abs_err, double &max_rel_err) {
    max_abs_err = 0.0;
    max_rel_err = 0.0;
    bool pass = true;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double ref = C_ref[i * N + j];
            double gpu = C_gpu[i * N + j];
            double abs_err = std::fabs(gpu - ref);
            double rel_err = abs_err / (std::fabs(ref) + 1e-6);
            max_abs_err = std::max(max_abs_err, abs_err);
            max_rel_err = std::max(max_rel_err, rel_err);
            if (abs_err > 1e-5 + 1e-4 * std::fabs(ref)) {
                pass = false;
            }
        }
    }
    return pass;
}

// 运行一组尺寸: do_verify = true 时与 CPU 参考对比并打印校验结果，否则仅测性能
static void run_case(int M, int N, int K, bool do_verify) {
    if (K % 4 != 0 || N % 4 != 0) {
        std::cerr << "[skip] 需要 K%4==0 且 N%4==0，跳过 M=" << M
                  << " N=" << N << " K=" << K << std::endl;
        return;
    }

    const size_t bytes_A = static_cast<size_t>(M) * K * sizeof(float);
    const size_t bytes_B = static_cast<size_t>(K) * N * sizeof(float);
    const size_t bytes_C = static_cast<size_t>(M) * N * sizeof(float);

    // 1. 主机端随机数据（固定种子，结果可复现）
    static std::mt19937 gen(42);
    static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> h_A(M * K), h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f), h_ref(M * N, 0.0f);
    for (float &v : h_A) v = dist(gen);
    for (float &v : h_B) v = dist(gen);

    // 2. 分配设备内存并拷贝输入
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    auto launch = [&]() {
        launch_sgemm_thread_tiling(d_A, d_B, d_C, M, N, K, grid, 256);
    };

    // 3. 预热，消除驱动懒加载与省电唤醒延迟
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 4. 计时（多次执行取平均）
    const int iterations = do_verify ? 3 : 20;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        launch();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iterations;

    // 5. 回拷结果并校验
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));
    bool pass = true;
    double max_abs_err = 0.0, max_rel_err = 0.0;
    if (do_verify) {
        sgemm_ref_cpu(h_A, h_B, h_ref, M, N, K);
        pass = verify(h_ref, h_C, M, N, max_abs_err, max_rel_err);
    }

    // 6. 打印报告
    double gflops = 2.0 * static_cast<double>(M) * N * K / (avg_ms * 1e-3) / 1e9;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "M=" << M << "  N=" << N << "  K=" << K
              << "   (" << bytes_C / (1024.0 * 1024.0) << " MB C)" << std::endl;
    std::cout << "Grid 配置   : " << grid.x << "x" << grid.y << " Blocks, "
              << BLOCK_SIZE << " Threads/Block" << std::endl;
    if (do_verify) {
        std::cout << "结果验证    : " << (pass ? "通过 (PASS)" : "失败 (FAIL)")
                  << "   (max abs err=" << max_abs_err
                  << ", max rel err=" << max_rel_err << ")" << std::endl;
    }
    std::cout << "平均计算耗时: " << avg_ms << " ms (x" << iterations << ")" << std::endl;
    std::cout << "性能        : " << gflops << " GFLOPS" << std::endl;

    // 7. 释放资源
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

void testSGEMM() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << "  (SM " << prop.major << "." << prop.minor
              << ", " << prop.multiProcessorCount << " SMs)" << std::endl;

    // 1. 正确性校验：覆盖边界情况
    const int cases[][3] = {
        {130, 140, 24},     // M、N 非 128 倍数；K 为 4 倍数、非 8 倍数
        {256, 256, 256},    // 规整尺寸
        {1037, 1036, 1036}, // M 非 8 倍数（余行）；N、K 非 128 倍数
        {1000, 1008, 1040}, // 大尺寸非规整
    };
    for (const auto &c : cases) {
        run_case(c[0], c[1], c[2], /*do_verify=*/true);
    }

    // 2. 性能基准
    run_case(4096, 4096, 4096, /*do_verify=*/false);
}
