/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once



inline __device__ int bfe(int i, int k)
{
    return (i >> k) & 1;
}


template <typename T, unsigned int SIZE = 32>
inline __device__ T shuffleSwapCompare(T x, int mask, int direction)
{
    auto y = __shfl_xor_sync(0xFFFFFFFF, x, mask, SIZE);
    return x < y == direction ? y : x;
}


template <typename T>
inline __device__ T bitonicSortStage(T v, unsigned int stage, unsigned int l)
{
    for (int i = stage; i >= 0; --i)
    {
        auto distance = 1 << i;
        unsigned int direction;

        direction = bfe(l, i) ^ bfe(l, stage + 1);
        v         = shuffleSwapCompare(v, distance, direction);
    }
    return v;
}

template <typename T>
inline __device__ T bitonicWarpSort(T v, unsigned int l)
{
    for (int stage = 0; stage < 5; ++stage)
    {
        v = bitonicSortStage(v, stage, l);
    }
    return v;
}
