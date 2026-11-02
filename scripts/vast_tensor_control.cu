// Synthetic tensor-activity control, not a performance benchmark.
// Build in the already authorized CUDA 12.8.1 devel image:
//   nvcc -O2 -std=c++17 vast_tensor_control.cu -lcublas -o tensor_control
// Run with a 30-second outer timeout. No models, data or random inputs.
// Contract: experiments/2026-11-01-vast-host-check.md, expectation 7.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_OK(call) do { \
    cudaError_t result = (call); \
    if (result != cudaSuccess) { \
        std::fprintf(stderr, "CUDA failure line=%d status=%s\n", __LINE__, cudaGetErrorString(result)); \
        std::exit(2); \
    } \
} while (0)

#define BLAS_OK(call) do { \
    cublasStatus_t result = (call); \
    if (result != CUBLAS_STATUS_SUCCESS) { \
        std::fprintf(stderr, "cuBLAS failure line=%d status=%d\n", __LINE__, int(result)); \
        std::exit(3); \
    } \
} while (0)

constexpr int N = 2048;
constexpr size_t elements = size_t(N) * N;
constexpr int batch_size = 64;
constexpr float target_window_ms = 5000.0f;

static long long unix_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

static size_t mismatches(const std::vector<float>& output) {
    size_t count = 0;
    // Every dot product is a sum of 2,048 exact 1*1 terms. 2,048 is exact FP32.
    for (float value : output) count += (value != 2048.0f);
    return count;
}

int main() {
    CUDA_OK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));
    cublasHandle_t handle{};
    BLAS_OK(cublasCreate(&handle));
    BLAS_OK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    int blas_version = 0;
    BLAS_OK(cublasGetVersion(handle, &blas_version));

    __half *a = nullptr, *b = nullptr;
    float* c = nullptr;
    CUDA_OK(cudaMalloc(&a, elements * sizeof(__half)));
    CUDA_OK(cudaMalloc(&b, elements * sizeof(__half)));
    CUDA_OK(cudaMalloc(&c, elements * sizeof(float)));
    const std::vector<__half> ones(elements, __float2half(1.0f));
    std::vector<float> output(elements);
    CUDA_OK(cudaMemcpy(a, ones.data(), elements * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(b, ones.data(), elements * sizeof(__half), cudaMemcpyHostToDevice));
    const float alpha = 1.0f, beta = 0.0f;
    auto gemm = [&]() {
        BLAS_OK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            N, N, N, &alpha,
                            a, CUDA_R_16F, N, b, CUDA_R_16F, N,
                            &beta, c, CUDA_R_32F, N,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };

    std::printf("device=0 name=%s cublas_version=%d N=%d elements=%zu expected=2048\n",
                prop.name, blas_version, N, elements);
    std::puts("tensor_operations=requested_not_independently_confirmed input=FP16 output=FP32 compute=FP32");
    gemm();
    gemm();
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(output.data(), c, elements * sizeof(float), cudaMemcpyDeviceToHost));
    const size_t warmup_bad = mismatches(output);
    std::printf("warmup_gemms=2 warmup_mismatches=%zu\n", warmup_bad);
    if (warmup_bad) return 4;

    // Validate that the exact comparator rejects a deliberate one-element error.
    output[0] = 2049.0f;
    const size_t negative_bad = mismatches(output);
    std::printf("deliberate_corruption_mismatches=%zu expected_corruption_mismatches=1\n", negative_bad);
    if (negative_bad != 1) return 5;

    cudaEvent_t start{}, end{};
    CUDA_OK(cudaEventCreate(&start));
    CUDA_OK(cudaEventCreate(&end));
    const auto wall_start = std::chrono::steady_clock::now();
    std::printf("control_window_start_unix_ms=%lld\n", unix_ms());
    std::fflush(stdout);
    CUDA_OK(cudaEventRecord(start));
    float elapsed_ms = 0.0f;
    uint64_t submitted = 0;
    do {
        for (int i = 0; i < batch_size; ++i) gemm();
        CUDA_OK(cudaEventRecord(end));
        CUDA_OK(cudaEventSynchronize(end));
        CUDA_OK(cudaEventElapsedTime(&elapsed_ms, start, end));
        submitted += batch_size;
        if (std::chrono::steady_clock::now() - wall_start > std::chrono::seconds(12)) {
            std::fprintf(stderr, "Control window host guard exceeded; outer timeout remains required.\n");
            return 6;
        }
    } while (elapsed_ms < target_window_ms);
    CUDA_OK(cudaDeviceSynchronize());
    std::printf("control_window_end_unix_ms=%lld submitted_control_gemms=%llu control_window_reached=PASS\n",
                unix_ms(), static_cast<unsigned long long>(submitted));
    CUDA_OK(cudaMemcpy(output.data(), c, elements * sizeof(float), cudaMemcpyDeviceToHost));
    const size_t final_bad = mismatches(output);
    std::printf("final_mismatches=%zu checked_elements=%zu\n", final_bad, elements);

    CUDA_OK(cudaEventDestroy(end));
    CUDA_OK(cudaEventDestroy(start));
    BLAS_OK(cublasDestroy(handle));
    CUDA_OK(cudaFree(c));
    CUDA_OK(cudaFree(b));
    CUDA_OK(cudaFree(a));
    if (final_bad) return 7;
    std::puts("SYNTHETIC_GEMM_EXACT_PASS; assess external counter response separately; no performance claim");
    return 0;
}
