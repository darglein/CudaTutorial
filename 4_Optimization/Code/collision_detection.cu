/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Timer.h"

#include <iostream>

#include "tiny-eigen/matrix.h"
#include <cuda_profiler_api.h>
#include <thrust/device_vector.h>

using vec3               = Eigen::Matrix<float, 3, 1>;
const int MAX_COLLISIONS = 100000;

__device__ int GlobalThreadId()
{
    return blockIdx.x * gridDim.x + threadIdx.x;
}
int iDivUp(int a, int b)
{
    return (a + b - (1)) / b;
}

struct Particle
{
    vec3 position;
    float radius;
};

__host__ __device__ bool Collide(const Particle& p1, const Particle& p2)
{
    float r2 = p1.radius + p2.radius;
    return (p1.position - p2.position).squaredNorm() < r2 * r2;
}

// =======================================================================================

__global__ void RedBlueParticleCollisionSimple(Particle* particles1, Particle* particles2, int n, int m, int2* list,
                                               int* counter)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= n || j >= m) return;


    const Particle& p1 = particles1[i];
    const Particle& p2 = particles2[j];


    if (Collide(p1, p2))
    {
        int index = atomicAdd(counter, 1);
        if (index < MAX_COLLISIONS)
        {
            list[index] = {i, j};
        }
    }
}


template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y, int K>
__global__ void RedBlueParticleCollisionShared(Particle* particles1, Particle* particles2, int n, int m, int2* list,
                                               int* counter)
{
    __shared__ Particle shared_particles1[16 * K];
    __shared__ Particle shared_particles2[16 * K];

    int block_i = blockIdx.x * 16 * K;
    int block_j = blockIdx.y * 16 * K;
    // int tid     = threadIdx.x + threadIdx.y * blockDim.x;

    if (block_i >= n || block_j >= m) return;

    // Load to smem
    if (threadIdx.y < K)
    {
        static_assert(K <= BLOCK_SIZE_X && K <= BLOCK_SIZE_Y, "a");
        int offset                = threadIdx.x + threadIdx.y * 16;
        shared_particles1[offset] = (particles1)[block_i + offset];
        shared_particles2[offset] = (particles2)[block_j + offset];
    }

    __syncthreads();

    for (int k = 0; k < K; ++k)
    {
        for (int l = 0; l < K; ++l)
        {
            int local_i = k * blockDim.x + threadIdx.x;
            int local_j = l * blockDim.y + threadIdx.y;

            int i = block_i + local_i;
            int j = block_j + local_j;

            if (i >= n || j >= m) continue;

            const Particle& p1 = shared_particles1[local_i];
            const Particle& p2 = shared_particles2[local_j];

            if (Collide(p1, p2))
            {
                int index = atomicAdd(counter, 1);
                if (index < MAX_COLLISIONS)
                {
                    list[index] = {i, j};
                }
            }
        }
    }
}

template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y, int K>
__global__ void RedBlueParticleCollisionSharedOptimized(Particle* particles1, Particle* particles2, int n, int m,
                                                        int2* list, int* counter)
{
    __shared__ float4 shared_particles1[16 * K];
    __shared__ float4 shared_particles2[16 * K];


    int block_i = blockIdx.x * 16 * K;
    int block_j = blockIdx.y * 16 * K;
    //    int tid     = threadIdx.x + threadIdx.y * blockDim.x;

    if (block_i >= n || block_j >= m) return;

    // Load to smem
    if (threadIdx.y < K)
    {
        static_assert(K <= BLOCK_SIZE_X && K <= BLOCK_SIZE_Y, "a");
        int offset                = threadIdx.x + threadIdx.y * 16;
        shared_particles1[offset] = ((float4*)particles1)[block_i + offset];
        shared_particles2[offset] = ((float4*)particles2)[block_j + offset];
    }



    __syncthreads();


    for (int k = 0; k < K * (16 / BLOCK_SIZE_X); ++k)
    {
        int local_i = k * blockDim.x + threadIdx.x;
        int i       = block_i + local_i;
        Particle p1 = ((Particle*)shared_particles1)[local_i];

        for (int l = 0; l < K * (16 / BLOCK_SIZE_Y); ++l)
        {
            int local_j = l * blockDim.y + threadIdx.y;
            int j       = block_j + local_j;
            Particle p2 = ((Particle*)shared_particles2)[local_j];
            if (j >= m | i >= n) p2.position.x() = std::numeric_limits<float>::quiet_NaN();


            if (Collide(p1, p2))
            {
                int index = atomicAdd(counter, 1);
                if (index < MAX_COLLISIONS)
                {
                    list[index] = {i, j};
                }
            }
        }
    }
}

int main(int argc, char* argv[])
{
    int n = 3940;
    int m = 5824;

    const int K     = 4;
    const int steps = 11;

    const int block_size_x = 16;
    const int block_size_y = 16;

    std::vector<Particle> particles1(n);
    std::vector<Particle> particles2(m);

    srand(1056735);
    auto rand_float = []() { return ((rand() % 10000) / 10000.f) * 2 - 1; };
    for (Particle& p : particles1)
    {
        p.position = vec3(rand_float(), rand_float(), rand_float()) * 25;
        p.radius   = 1;
    }
    for (Particle& p : particles2)
    {
        p.position = vec3(rand_float(), rand_float(), rand_float()) * 25;
        p.radius   = 1;
    }

    thrust::device_vector<Particle> d_particles1(particles1);
    thrust::device_vector<Particle> d_particles2(particles2);

    // Add padding to simplify shared memory kernel
    d_particles1.resize(n + 16 * K);
    d_particles2.resize(m + 16 * K);

    thrust::device_vector<int2> d_collision_list(MAX_COLLISIONS);
    thrust::device_vector<int> d_collision_count(1, 0);
    d_collision_count[0] = 0;


    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    cudaProfilerStart();


    float time_simple = measureObject2(
        steps, [&]() { d_collision_count[0] = 0; },
        [&]()
        {
            int blocks_x = iDivUp(n, block_size_x);
            int blocks_y = iDivUp(m, block_size_y);
            RedBlueParticleCollisionSimple<<<dim3(blocks_x, blocks_y, 1), dim3(block_size_x, block_size_y, 1)>>>(
                d_particles1.data().get(), d_particles2.data().get(), n, m, d_collision_list.data().get(),
                d_collision_count.data().get());
        });
    std::cout << "RedBlueParticleCollisionSimple            " << time_simple << " ms." << std::endl;
    int num_collisions_simple = d_collision_count[0];

#if 1
    float time_shared = measureObject2(
        steps, [&]() { d_collision_count[0] = 0; },
        [&]()
        {
            int blocks_x = iDivUp(n, block_size_x * K);
            int blocks_y = iDivUp(m, block_size_y * K);
            RedBlueParticleCollisionShared<block_size_x, block_size_y, K>
                <<<dim3(blocks_x, blocks_y, 1), dim3(block_size_x, block_size_y, 1)>>>(
                    d_particles1.data().get(), d_particles2.data().get(), n, m, d_collision_list.data().get(),
                    d_collision_count.data().get());
        });
    std::cout << "RedBlueParticleCollisionShared            " << time_shared << " ms." << std::endl;
#endif
    int num_collisions_shared = d_collision_count[0];

#if 1
    float time_shared_optimized = measureObject2(
        steps, [&]() { d_collision_count[0] = 0; },
        [&]()
        {
            int blocks_x = iDivUp(n, block_size_x * K);
            int blocks_y = iDivUp(m, block_size_y * K);
            RedBlueParticleCollisionSharedOptimized<block_size_x, block_size_y, K>
                <<<dim3(blocks_x, blocks_y, 1), dim3(block_size_x, block_size_y, 1)>>>(
                    d_particles1.data().get(), d_particles2.data().get(), n, m, d_collision_list.data().get(),
                    d_collision_count.data().get());
        });
    std::cout << "RedBlueParticleCollisionSharedOptimized   " << time_shared_optimized << " ms." << std::endl;
#endif
    int num_collisions_shared_optimized = d_collision_count[0];

    cudaProfilerStop();
    cudaDeviceSynchronize();

    std::atomic_int num_collisions_cpu;
    num_collisions_cpu = 0;
    float time_cpu;
    {
        ScopedTimer<float> timer(time_cpu);
#pragma omp parallel for
        for (int i = 0; i < particles1.size(); ++i)
        {
            auto& p1 = particles1[i];
            for (auto& p2 : particles2)
            {
                if (Collide(p1, p2))
                {
                    num_collisions_cpu++;
                }
            }
        }
    }
    std::cout << "RedBlueParticleCollision CPU              " << time_cpu << " ms" << std::endl;
    std::cout << std::endl;
    std::cout << "Num Collisions GPU: " << num_collisions_simple << " " << num_collisions_shared << " "
              << num_collisions_shared_optimized << std::endl;
    std::cout << "Num Collisions CPU: " << num_collisions_cpu << std::endl;

    return 0;
}
