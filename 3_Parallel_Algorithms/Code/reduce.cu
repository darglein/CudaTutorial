/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Eigen/Core"
#include "reduce.h"

#include <iomanip>
#include <iostream>

#include <thrust/device_vector.h>

int iDivUp(int a, int b)
{
    return (a + b - (1)) / b;
}

__global__ static void WarpReduce(int* data, int N, int* output)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int v   = tid >= N ? 0 : data[tid];
    v       = warpReduce(v, thrust::plus<int>());

    if (tid == 0) output[0] = v;
}


__global__ static void BlockReduce(int* data, int N, int* output)
{
    __shared__ int blockSum;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int v   = tid >= N ? 0 : data[tid];
    v       = blockReduceSumAtomic(v, blockSum);
    if (threadIdx.x == 0) output[0] = v;
}

__global__ static void Reduce(int* data, int N, int* output)
{
    __shared__ int blockSum;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int v   = tid >= N ? 0 : data[tid];
    v       = blockReduceSumAtomic(v, blockSum);
    if (threadIdx.x == 0)
    {
        atomicAdd(output, v);
    }
}

int main(int argc, char* argv[])
{
    int N = 2467173;
    std::vector<int> data(N);
    for (auto& d : data)
    {
        d = rand() % 4;
    }
    thrust::device_vector<int> d_data = data;
    thrust::device_vector<int> d_output(1);



    {
        // Warp reduce
        int n = 32;
        thrust::device_vector<int> part(d_data.begin(), d_data.begin() + n);
        int reference = thrust::reduce(part.begin(), part.end());

        d_output[0] = 0;
        WarpReduce<<<1, 32>>>(part.data().get(), n, d_output.data().get());
        int sum = d_output[0];
        std::cout << "Our Warp Reduce:   " << std::setw(8) << sum << "   thrust::reduce: " << std::setw(8) << reference
                  << std::endl;
    }

    {
        // Block reduce
        int n = 256;
        thrust::device_vector<int> part(d_data.begin(), d_data.begin() + n);
        int reference = thrust::reduce(part.begin(), part.end());

        d_output[0] = 0;
        BlockReduce<<<1, 256>>>(part.data().get(), n, d_output.data().get());
        int sum = d_output[0];
        std::cout << "Our Block Reduce:  " << std::setw(8) << sum << "   thrust::reduce: " << std::setw(8) << reference
                  << std::endl;
    }

    {
        // Global reduce
        auto part     = d_data;
        int reference = thrust::reduce(part.begin(), part.end());

        d_output[0] = 0;
        Reduce<<<iDivUp(N, 256), 256>>>(part.data().get(), N, d_output.data().get());
        int sum = d_output[0];
        std::cout << "Our Global Reduce: " << std::setw(8) << sum << "   thrust::reduce: " << std::setw(8) << reference
                  << std::endl;
    }

    return 0;
}
