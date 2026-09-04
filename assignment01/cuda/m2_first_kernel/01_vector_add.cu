// Question 2.1: Vector Addition (Fill-in-the-blank)
// Each of the six blanks tests one concept. Complete the blanks, compile and run the program, and you will see "PASS" if successful.
// This file cannot be compiled before all blanks are filled.
#include "common.h"

//
__global__ void vectorAdd(const float *a, const float *b, float *c, int n)
{
    //
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    //
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

int main()
{
    const int n = 1000003; // Intentionally pick a number that is not an integer multiple of 256
    size_t bytes = (size_t)n * sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    fill_random(h_a, n, 1);
    fill_random(h_b, n, 2);
    for (int i = 0; i < n; i++)
        h_ref[i] = h_a[i] + h_b[i];

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    //
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    //
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    //
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    REPORT(check_close(h_c, h_ref, n));
    return 0;
}
