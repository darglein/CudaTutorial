/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "tiny-eigen/matrix.h"
#include "reduce.h"

#include <algorithm>
#include <iomanip>
#include <iostream>

#include <thrust/device_vector.h>

using vec3 = Eigen::Matrix<float, 3, 1>;

int iDivUp(int a, int b)
{
    return (a + b - (1)) / b;
}

struct OurReduceOp
{
    __host__ __device__ int operator()(int a, int b)
    {
        if (values[a].squaredNorm() > values[b].squaredNorm())
        {
            return a;
        }
        else
        {
            return b;
        }
    }
    vec3* values;
};


template <int BLOCK_SIZE, typename OP>
__global__ static void Reduce(int* data, int N, int* output, OP op)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int v   = tid >= N ? 0 : data[tid];
    v       = blockReduce<BLOCK_SIZE>(v, op);
    if (threadIdx.x == 0)
    {
        // Non-blocking custom reduce with CAS
        // (only works if the reduced value is 4 or 8 bytes)
        int old = *output, assumed;
        do
        {
            assumed     = old;
            int reduced = op(v, assumed);
            old         = atomicCAS(output, assumed, reduced);
        } while (assumed != old);
    }
}

int main(int argc, char* argv[])
{
    int N = 1232563;
    std::vector<int> keys(N);
    std::vector<vec3> values(N);

    auto rand_float = []() { return ((rand() % 10000) / 10000.f) * 2 - 1; };
    for (int i = 0; i < N; ++i)
    {
        keys[i] = i;
        values[i] = vec3(rand_float(), rand_float(), rand_float());
    }
    thrust::device_vector<int> d_keys               = keys;
    thrust::device_vector<vec3> d_values = values;
    thrust::device_vector<int> d_output(1);
    d_output[0] = 0;

    OurReduceOp op;
    op.values = d_values.data().get();
    //    auto op = thrust::plus<int>();

    int reference = thrust::reduce(d_keys.begin(), d_keys.end(), 0, op);

    Reduce<256><<<iDivUp(N, 256), 256>>>(d_keys.data().get(), N, d_output.data().get(), op);
    int result = d_output[0];

    //    std::cout << "Our Global Reduce: " << std::setw(8) << result << "   thrust::reduce: " << std::setw(8) <<
    //    reference
    //              << std::endl;

    std::cout << "Our Global Reduce: " << std::setw(8) << result << " " << values[result].squaredNorm()
              << "   thrust::reduce: " << std::setw(8) << reference << " " << values[reference].squaredNorm()
              << std::endl;


    return 0;
}
