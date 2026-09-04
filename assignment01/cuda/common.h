// Small tools shared by all exercises in this directory. You are required to write the final challenge problem (Question 2.9) from scratch.
// Do not include this file when working on that problem, due to the error checking and timing functions contained herein.
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Wrap each CUDA API call, and report the file name, line number and cause immediately once an error occurs.
#define CUDA_CHECK(call)                                        \
    do                                                          \
    {                                                           \
        cudaError_t err_ = (call);                              \
        if (err_ != cudaSuccess)                                \
        {                                                       \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n",     \
                    cudaGetErrorName(err_), __FILE__, __LINE__, \
                    cudaGetErrorString(err_));                  \
            exit(1);                                            \
        }                                                       \
    } while (0)

// The kernel startup itself has no return value, and these two lines of code are used to check for related errors.
#define CUDA_CHECK_KERNEL()                  \
    do                                       \
    {                                        \
        CUDA_CHECK(cudaGetLastError());      \
        CUDA_CHECK(cudaDeviceSynchronize()); \
    } while (0)

// Timer based on cudaEvent, measuring the time consumption on the GPU (in milliseconds).
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

// Pseudorandom filling with fixed seed to ensure consistent data in each run.
static inline void fill_random(float *p, long n, unsigned seed = 42)
{
    srand(seed);
    for (long i = 0; i < n; i++)
        p[i] = (float)(rand() % 1000) / 100.0f;
}

// Conduct element-wise comparison. If the relative error exceeds eps, it is deemed a failure. Return 1 to indicate success.
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

// ---------------- Performance Comparison ----------------
// Print the ratio of the two versions. An extra prompt will be printed if the ratio is lower than warn_below, while the exit code remains unchanged.
// The test only displays the figures; you need to judge speed by yourself with reference to the explanations on the handout.
// If warn_below 0, no speedup is expected for this problem, and no prompt will be printed.
static inline float report_speedup(const char *label, float base_ms,
                                   float opt_ms, float warn_below,
                                   const char *hint)
{
    float ratio = opt_ms > 0.f ? base_ms / opt_ms : 0.f;
    printf("%s = %.2fx\n", label, ratio);
    if (warn_below > 0.f && ratio < warn_below)
    {
        printf("WARN: %s（不影响 PASS）\n", hint);
    }
    return ratio;
}

// ---------------- Machine-readable result lines ----------------
// For consumption by external evaluation frameworks. Not printed by default.
// Output is generated only when the environment variable WMHPC_RESULT=1 is set, keeping outputs clean during regular runs.
// Convention: Exit codes only indicate correctness; performance metrics never affect exit codes.
static inline void emit_result(const char *prob, const char *status,
                               const char *metrics_json)
{
    const char *on = getenv("WMHPC_RESULT");
    if (!on || on[0] == '0' || on[0] == '\0')
        return;
    if (!metrics_json)
        metrics_json = "{}";

    int dev = 0;
    cudaDeviceProp prop;
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
    {
        printf("##RESULT {\"prob\":\"%s\",\"status\":\"%s\",\"metrics\":%s,"
               "\"device\":\"%s\",\"sm\":%d,\"cc\":\"%d.%d\"}\n",
               prob, status, metrics_json, prop.name,
               prop.multiProcessorCount, prop.major, prop.minor);
    }
    else
    {
        printf("##RESULT {\"prob\":\"%s\",\"status\":\"%s\",\"metrics\":%s}\n",
               prob, status, metrics_json);
    }
    fflush(stdout);
}
