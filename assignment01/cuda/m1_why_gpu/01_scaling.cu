// Question 1.5: Parallelism and throughput. Run the same vector addition in four different ways.
// Compare the time consumption of each method and explain the discrepancies.
// Compile and run: make run/m1_why_gpu/01_scaling
#include <chrono>
#include "common.h"

// single thread
__global__ void add_one_thread(const float *a, const float *b, float *c, int n)
{
    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

// one block 256 thread
__global__ void add_one_block(const float *a, const float *b, float *c, int n)
{
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        c[i] = a[i] + b[i];
}

// grid
__global__ void add_grid(const float *a, const float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}

int main()
{
    const int n = 1 << 22; // 4M
    size_t bytes = (size_t)n * sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    fill_random(h_a, n, 1);
    fill_random(h_b, n, 2);

    // CPU single thread
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; i++)
        h_ref[i] = h_a[i] + h_b[i];
    auto t1 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("CPU single thread      : %10.3f ms  (%6.2f ns/element)\n", cpu_ms,
           cpu_ms * 1e6 / n);

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // warm up
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_grid<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK_KERNEL();

    GpuTimer timer;

    timer.start();
    add_one_thread<<<1, 1>>>(d_a, d_b, d_c, n);
    float ms1 = timer.stop_ms();
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    if (!check_close(h_c, h_ref, n))
        REPORT(0);
    printf("GPU <<<1, 1>>>  : %10.3f ms  (%6.2f ns/element)\n", ms1, ms1 * 1e6 / n);

    timer.start();
    add_one_block<<<1, 256>>>(d_a, d_b, d_c, n);
    float ms2 = timer.stop_ms();
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    if (!check_close(h_c, h_ref, n))
        REPORT(0);
    printf("GPU <<<1, 256>>>: %10.3f ms  (%6.2f ns/element)\n", ms2, ms2 * 1e6 / n);

    timer.start();
    add_grid<<<blocks, threads>>>(d_a, d_b, d_c, n);
    float ms3 = timer.stop_ms();
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    if (!check_close(h_c, h_ref, n))
        REPORT(0);
    printf("GPU grid   : %10.3f ms  (%6.2f ns/element, %d blocks x %d threads)\n",
           ms3, ms3 * 1e6 / n, blocks, threads);

    REPORT(1);
    return 0;
}
