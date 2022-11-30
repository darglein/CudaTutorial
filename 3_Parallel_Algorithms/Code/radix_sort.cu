/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include <iostream>
#include <algorithm>

#include "bitonic_sort.h"
#include <thrust/device_vector.h>


template <bool one>
struct GetBitOp
{
    int k;
    GetBitOp(int k) : k(k) {}
    __device__ inline int operator()(int a) { return ((a >> k) & 1) == one; }
};

static void radixSortHelper(thrust::device_vector<int>& d, thrust::device_vector<int>& p, thrust::device_vector<int>& s,
                            thrust::device_vector<int>& t, int bit)
{
#if 1
    // Implementation with scan+scatter

    // Compute predicate array for 0-bits
    thrust::transform(d.begin(), d.end(), p.begin(), GetBitOp<false>(bit));

    // Scan over the predicate array and store it in s
    thrust::exclusive_scan(p.begin(), p.end(), s.begin(), 0);

    // Write all 0-bit integers to the scanned positions
    // This writes only if the predicate also evaluates to true
    thrust::scatter_if(d.begin(), d.end(), s.begin(), p.begin(), t.begin());

    // Total number of 0 bits
    int count = thrust::reduce(p.begin(), p.end());

    // Same with 1-bit integers, but use 'count' as the initial value in the scan
    thrust::transform(d.begin(), d.end(), p.begin(), GetBitOp<true>(bit));
    thrust::exclusive_scan(p.begin(), p.end(), s.begin(), count);
    thrust::scatter_if(d.begin(), d.end(), s.begin(), p.begin(), t.begin());
#else
    // Implementation with copy_if
    auto it = thrust::copy_if(d.begin(), d.end(), t.begin(), GetBitOp<false>(bit));
    thrust::copy_if(d.begin(), d.end(), it, GetBitOp<true>(bit));
#endif

    // Both variants don't work inplace!
    thrust::copy(t.begin(), t.end(), d.begin());
}

static void radixSort(thrust::device_vector<int>& data)
{
    int N = data.size();

    // Temporary arrays
    thrust::device_vector<int> pred(N);
    thrust::device_vector<int> scan(N);
    thrust::device_vector<int> temp(N);

    // Sort from least to most significant bit
    for (int i = 0; i < 32; ++i) radixSortHelper(data, pred, scan, temp, i);
}

int main(int argc, char* argv[])
{
    int N   = 1024 * 1024;
    using T = int;
    thrust::host_vector<T> h_data(N);

    // Initialize with random values
    for (auto& f : h_data)
    {
        f = abs(rand());
    }


    std::cout << "Sorting " << N << " elements..." << std::endl;
    thrust::device_vector<T> d_data = h_data;
    radixSort(d_data);

    thrust::host_vector<T> res = d_data;
    if (std::is_sorted(res.begin(), res.end()))
    {
        std::cout << "Success! All elements are in the correct order!" << std::endl;
    }
    else
    {
        std::cout << "Sort failed!" << std::endl;
    }

    return 0;
}
