
#include "common.h"

/*
bank 是 shared memory 的物理存储组织，类似硬盘的 RAID striping：

shared memory 物理上 = 32 个独立的 SRAM slice（各 4 字节宽）
                     ├── bank 0: 地址 0, 128, 256, ...
                     ├── bank 1: 地址 4, 132, 260, ...
                     ├── ...
                     └── bank 31: 地址 124, 252, 380, ...

每个 bank 有自己的读写总线。当 warp 的 32 个线程同时发出访问请求时：

- 32 个请求打到 32 个不同的 bank → 32 条总线同时工作 → 1 cycle 完成
- 2 个请求打到 同一个 bank 的不同地址 → 1 条总线要服务 2 个请求 → 串行 2 cycle

bank 不决定"谁能访问"，只是决定"谁和谁抢同一条总线"。所有线程都能访问任意地址，只是抢同一个 bank 时变慢, 成为bank conflict*/

int main()
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("GPU             : %s\n", prop.name);                     // NVIDIA RTX PRO 6000 Blackwell Server Edition
    printf("compute capability  : %d.%d\n", prop.major, prop.minor); // 12.0

    // ======  1 ======
    printf("SM              : %d\n", /* todo */ prop.multiProcessorCount); // 188

    // ====== 2 size ======
    printf("warp size           : %d\n", /* todo */ prop.warpSize); // 32

    // ====== 3 per block  shared memory======
    printf("shared mem / block  : %zu\n", (size_t)/* todo */ prop.sharedMemPerBlock); // 49152

    // ======  4: ======
    printf("max threads / SM    : %d\n", /* todo */ prop.maxThreadsPerMultiProcessor); // 1536

    // ====== 5 ======
    printf("global mem          : %zu\n", (size_t)/* todo */ prop.totalGlobalMem); // 101975851008

    printf("max threads / block : %d\n", prop.maxThreadsPerBlock); // 1024
    return 0;
}

/*

GPU             : NVIDIA B300 SXM6 AC
compute capability  : 10.3
SM              : 148
warp size           : 32
shared mem / block  : 49152
max threads / SM    : 2048
global mem          : 287428640768
max threads / block : 1024

*/