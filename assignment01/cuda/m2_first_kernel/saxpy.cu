#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cstring>
#include <iostream>

using namespace std;

#define CUDA_CHECK(call)                                                                                                        \
    do                                                                                                                          \
    {                                                                                                                           \
        cudaError_t err = (call);                                                                                               \
        if (err != cudaSuccess)                                                                                                 \
        {                                                                                                                       \
            fprintf(stderr, "CUDA error: %s at %s:%d\n%s", cudaGetErrorName(err), __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1);                                                                                                            \
        }                                                                                                                       \
    } while (0)

#define CUDA_CHECK_KERNEL()                  \
    do                                       \
    {                                        \
        CUDA_CHECK(cudaGetLastError());      \
        CUDA_CHECK(cudaDeviceSynchronize()); \
    } while (0)

#define REPORT(ok)            \
    do                        \
    {                         \
        if (ok)               \
        {                     \
            printf("PASS\n"); \
        }                     \
        else                  \
        {                     \
            printf("FAIL\n"); \
            exit(1);          \
        }                     \
    } while (0)

struct GpuTimer
{
    cudaEvent_t start_, stop_;
    GpuTimer()
    {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer()
    {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    float stop_ms()
    {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};
static inline int check_close(const float *got, const float *want, long n,
                              float eps = 1e-4f)
{
    for (long i = 0; i < n; i++)
    {
        if (fabsf(got[i] - want[i]) > eps * (1.0f + fabsf(want[i])))
        {
            fprintf(stderr, "MISMATCH at %ld: got %f, want %f\n", i,
                    (double)got[i], (double)want[i]);
            return 0;
        }
    }
    return 1;
}

__global__ void saxpy(const float *x, float *y, long n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        y[idx] = 2.0f * x[idx] + y[idx];
    }
}

int main(int argc, char **argv)
{
    long n = strtol(argv[1], NULL, 10);
    if (n == 0)
    {
        printf("SUM=0\n");
        exit(0);
    }
    float *a = (float *)malloc(n * sizeof(float));
    float *b = (float *)malloc(n * sizeof(float));
    float *c = (float *)malloc(n * sizeof(float));
    for (long i = 0; i < n; i++)
    {
        a[i] = ((i % 2048) - 1024) * 0.5f;
        b[i] = (i % 1024) - 512;
        c[i] = b[i] + 2.0f * a[i];
    }
    float *x = (float *)malloc(n * sizeof(float));
    float *y = (float *)malloc(n * sizeof(float));
    cudaMalloc(&x, n * sizeof(float));
    cudaMalloc(&y, n * sizeof(float));
    cudaMemcpy(x, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y, b, n * sizeof(float), cudaMemcpyHostToDevice);

    GpuTimer timer;
    timer.start();
    saxpy<<<(n + 255) / 256, 256>>>(x, y, n);
    timer.stop_ms();
    // printf("GPU SAXPY: %.3f ms\n", timer.stop_ms());
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(b, y, n * sizeof(float), cudaMemcpyDeviceToHost));

    double SUM = 0.0;
    for (long i = 0; i < n; i++)
    {
        SUM += b[i];
    }
    printf("SUM=%.0f\n", SUM);

    return 0;
}