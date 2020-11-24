/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "thrust/functional.h"

template <typename T, typename OP>
__device__ inline T warpReduce(T val, OP op)
{
    static_assert(sizeof(T) <= 8, "Only 8 byte reductions are supportet.");
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
    {
        auto v = __shfl_down_sync(0xFFFFFFFF, val, offset);
        val    = op(val, v);
    }
    return val;
}


template <typename T>
__device__ inline T blockReduceSumAtomic(T val, T& blockSum)
{
    int lane = threadIdx.x & (32 - 1);

    // Each warp reduces with registers
    val = warpReduce(val, thrust::plus<T>());

    // Init shared memory
    if (threadIdx.x == 0) blockSum = T(0);

    __syncthreads();


    // The first thread in each warp writes to smem
    if (lane == 0)
    {
        atomicAdd(&blockSum, val);
    }

    __syncthreads();

    // The first thread in this block has the result
    // Optional: remove if so that every thread has the result
    if (threadIdx.x == 0) val = blockSum;

    return val;
}


template <int BLOCK_SIZE, typename T, typename OP>
__device__ inline T blockReduce(T val, OP op)
{
    __shared__ T shared[BLOCK_SIZE / 32];

    int lane   = threadIdx.x % 32;
    int warpid = threadIdx.x / 32;

    // Each warp reduces with registers
    val = warpReduce(val, op);

    // The first thread in each warp writes to smem
    if (lane == 0)
    {
        shared[warpid] = val;
    }

    __syncthreads();


    if (threadIdx.x < BLOCK_SIZE / 32)
    {
        val = shared[threadIdx.x];
    }
    else
    {
        val = 0;
    }

    if (warpid == 0)
    {
        val = warpReduce(val, op);
    }


    return val;
}
