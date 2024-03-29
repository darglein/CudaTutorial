﻿/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "tiny-eigen/matrix.h"
#include "Timer.h"

#include <iomanip>
#include <iostream>
#include <vector>

#include <cuda_profiler_api.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

inline int iDivUp(int a, int b)
{
    return (a + b - (1)) / b;
}
template <typename T>
using pinned_vector = ::thrust::host_vector<T, ::thrust::cuda::experimental::pinned_allocator<T>>;
using vec4 = Eigen::Matrix<float, 4, 1>;

template <int K>
class Element
{
   public:
    vec4 data;

    __host__ __device__ inline void operator()()
    {
        for (int k = 0; k < K * 512; ++k)
        {
            data = data * 3.1f + data;
        }
    }
};


template <typename T>
__global__ static void process(T* data, int N)
{
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N; tid += blockDim.x * gridDim.x)
    {
        T e = data[tid];
        e();
        data[tid] = e;
    }
}


template <int K>
static void uploadProcessDownloadAsync(int N, int slices, int streamCount)
{
    using T = Element<K>;
    pinned_vector<T> h_data(N);
    thrust::device_vector<T> d_data(N);

    int sliceN = N / slices;


    std::vector<cudaStream_t> streams(streamCount);
    for (auto& s : streams)
    {
        cudaStreamCreate(&s);
    }



    float time;
    {
        CudaScopedTimer timer(time);


        for (int i = 0; i < slices; ++i)
        {
            T* d_ptr = d_data.data().get() + i * sliceN;
            T* h_ptr = h_data.data() + i * sliceN;

            auto& stream = streams[i % streamCount];

            cudaMemcpyAsync(d_ptr, h_ptr, sliceN * sizeof(T), cudaMemcpyHostToDevice, stream);
            process<T><<<iDivUp(sliceN, 128), 128, 0, stream>>>(d_ptr, sliceN);
            cudaMemcpyAsync(h_ptr, d_ptr, sliceN * sizeof(T), cudaMemcpyDeviceToHost, stream);
        }
    }
    std::cout << "uploadProcessDownloadAsync Streams = " << std::setw(3) << streamCount << " Slices = " << std::setw(3)
              << slices << " Time: " << time << "ms." << std::endl;

    for (auto& s : streams)
    {
        cudaStreamDestroy(s);
    }
}

int main(int argc, char* argv[])
{
    cudaProfilerStart();
    uploadProcessDownloadAsync<2>(4 * 1024 * 1024, 1, 1);
    uploadProcessDownloadAsync<2>(4 * 1024 * 1024, 2, 2);
    uploadProcessDownloadAsync<2>(4 * 1024 * 1024, 4, 4);
    uploadProcessDownloadAsync<2>(4 * 1024 * 1024, 16, 4);
    cudaProfilerStop();
    std::cout << "Done." << std::endl;
}
