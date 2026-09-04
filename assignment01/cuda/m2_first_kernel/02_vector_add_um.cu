// Question 2.3: Convert explicit memory management to Unified Memory (MODIFY).
// Below is a fully runnable version with explicit memory management. Tasks:
//   0. Run it as-is once and record the execution time. This version will be overwritten by your modifications,
//      and it will serve as the benchmark for comparison in Step 4;
//   1. Replace cudaMalloc plus malloc with cudaMallocManaged;
//   2. Remove all cudaMemcpy calls. The kernel directly reads and writes the same set of pointers, and the CPU also performs direct reads;
//   3. Determine the positions where cudaDeviceSynchronize is required;
//   4. Compare the execution time of the two versions. The timing windows of both versions must be consistent: memory allocation and data filling are outside the timing window.
//      The window starts when "data is ready in memory" and ends when the CPU finishes reading all results
//      (the loop calculating an accumulated checksum below stands for "the CPU finishes reading all results"; do not delete this loop).
// The revised code must still PASS all tests.
#include <chrono>
#include "common.h"

__global__ void vectorAdd(const float *a, const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

int main()
{
    const int n = 1 << 24; // 16M
    size_t bytes = (size_t)n * sizeof(float);

    // First establish the CUDA context. The initial call to the CUDA API requires hundreds of milliseconds for initialization,
    // which would completely obscure the differences to be observed if included in the timing window.
    CUDA_CHECK(cudaFree(0));

    // Expected checksum, precomputed on the host and likewise excluded from timing.

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMallocManaged(&d_a, bytes));
    CUDA_CHECK(cudaMallocManaged(&d_b, bytes));
    CUDA_CHECK(cudaMallocManaged(&d_c, bytes));
    fill_random(d_a, n, 1);
    fill_random(d_b, n, 2);
    double want = 0;
    for (int i = 0; i < n; i++)
        want += (double)(d_a[i] + d_b[i]);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // =================time window start=================
    auto t0 = std::chrono::steady_clock::now();

    // No cudaMemcpy calls needed with Unified Memory

    vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK_KERNEL();

    // No cudaMemcpy calls needed with Unified Memory

    // CPU read the results. In the unified memory version, this step will trigger the migration of the result pages back to the host.
    double got = 0;
    for (int i = 0; i < n; i++)
        got += (double)d_c[i];

    auto t1 = std::chrono::steady_clock::now();
    // =================time window end=================

    printf("unified mem: %.1f ms\n",
           std::chrono::duration<double, std::milli>(t1 - t0).count());

    REPORT(fabs(got - want) <= 1e-3 * (1.0 + fabs(want)));
    return 0;
}
// cudaMemcpy:43.9 ms
// cudaMallocManaged: 484ms