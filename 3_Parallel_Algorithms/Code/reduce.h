/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


template <typename T>
__device__ inline T warpReduceSum(T val)
{
    for (int offset = 16; offset > 0; offset /= 2)
    {
        auto v = __shfl_down_sync(0xFFFFFFFF, val, offset);
        val    = val + v;
    }
    return val;
}


template <typename T>
__device__ inline T blockReduceSum(T val, T& blockSum)
{
    int lane = threadIdx.x & (32 - 1);

    // Each warp reduces with registers
    val = warpReduceSum(val);

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
