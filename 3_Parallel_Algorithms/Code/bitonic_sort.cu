/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Eigen/Core"

#include <iostream>

#include "bitonic_sort.h"
#include <thrust/device_vector.h>

__global__ static void WarpSort(int* data, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    auto v    = data[tid];
    v         = bitonicWarpSort(v, tid);
    data[tid] = v;
}



int main(int argc, char* argv[])
{
    int N = 32;
    std::vector<int> data(N);
    for (auto& d : data)
    {
        d = rand() % 100 - 50;
    }
    thrust::device_vector<int> d_data = data;

    std::cout << "Input (n=" << N << ")" << std::endl;
    std::cout << "{";
    for (auto a : data)
    {
        std::cout << a << ", ";
    }
    std::cout << "}" << std::endl;


    WarpSort<<<N / 32, 32>>>(d_data.data().get(), N);

    thrust::host_vector<int> res = d_data;

    std::cout << "Output: " << std::endl;
    std::cout << "{";
    for (auto a : res)
    {
        std::cout << a << ", ";
    }
    std::cout << "}" << std::endl;


    return 0;
}
