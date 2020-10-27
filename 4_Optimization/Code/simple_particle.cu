/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "Eigen/Core"
#include "Timer.h"

#include <iostream>
#include <vector>

#include <cuda_profiler_api.h>
#include <thrust/device_vector.h>

using vec3 = Eigen::Vector3f;

struct Particle
{
    vec3 position;
    float radius;
    vec3 velocity;
    float invMass;
};

struct EIGEN_ALIGN16 PositionRadius
{
    vec3 position;
    float radius;
};


struct EIGEN_ALIGN16 VelocityMass
{
    vec3 velocity;
    float invMass;
};

// ===== Helper functions ====
__device__ inline int GlobalThreadId()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
}
inline int iDivUp(int a, int b)
{
    return (a + b - (1)) / b;
}

// ===== Particle Integration Kernel ====
__global__ static void updateParticles(Particle* particles, int N, float dt)
{
    int tid = GlobalThreadId();
    if (tid >= N) return;
    Particle& p = particles[tid];
    p.position += p.velocity * dt;
    p.velocity += vec3(0, -9.81, 0) * dt;
}

__global__ static void updateParticles2(PositionRadius* prs, VelocityMass* vms, int N, float dt)
{
    int tid = GlobalThreadId();
    if (tid >= N) return;
    PositionRadius pr;
    VelocityMass vm;
    reinterpret_cast<int4*>(&pr)[0] = reinterpret_cast<int4*>(prs)[tid];
    reinterpret_cast<int4*>(&vm)[0] = reinterpret_cast<int4*>(vms)[tid];

    //    Particle& p = particles[tid];
    pr.position += vm.velocity * dt;
    vm.velocity += vec3(0, -9.81, 0) * dt;


    reinterpret_cast<int4*>(prs)[tid] = reinterpret_cast<int4*>(&pr)[0];
    reinterpret_cast<int4*>(vms)[tid] = reinterpret_cast<int4*>(&vm)[0];
    //    vms[tid] = vm;
    //    prs[tid] = pr;
}

int main(int argc, char* argv[])
{
    const int N     = 2500000;
    const int steps = 3;
    float dt        = 0.1;

    std::cout << "Simple Particle integration N = " << N << std::endl;

    // Allocate CPU and GPU memory
    std::vector<Particle> particles(N);
    thrust::device_vector<Particle> d_particles(N);

    thrust::device_vector<PositionRadius> d_pr(N);
    thrust::device_vector<VelocityMass> d_vm(N);

    // Initialize on the CPU
    for (Particle& p : particles)
    {
        //        p.position = vec3::Zero();
        //        p.velocity = vec3::Random();
    }

    // Upload memory
    //    thrust::copy(particles.begin(), particles.end(), d_particles.begin());

    cudaProfilerStart();
    // Integrate
    for (int i = 0; i < steps; ++i)
    {
        float time_gpu;
        {
            CudaScopedTimer timer(time_gpu);
            const int BLOCK_SIZE = 128;
            updateParticles<<<iDivUp(N, BLOCK_SIZE), BLOCK_SIZE>>>(d_particles.data().get(), N, dt);
        }
        std::cout << "Integrate  " << time_gpu << " ms." << std::endl;
    }

    std::cout << std::endl;


    for (int i = 0; i < steps; ++i)
    {
        float time_gpu;
        {
            CudaScopedTimer timer(time_gpu);
            const int BLOCK_SIZE = 128;
            updateParticles2<<<iDivUp(N, BLOCK_SIZE), BLOCK_SIZE>>>(d_pr.data().get(), d_vm.data().get(), N, dt);
        }
        std::cout << "Integrate2 " << time_gpu << " ms." << std::endl;
    }
    cudaProfilerStop();
    // Download
    thrust::copy(d_particles.begin(), d_particles.end(), particles.begin());

    //    for (Particle& p : particles)
    //    {
    //        std::cout << p.position.transpose() << " " << p.velocity.transpose() << std::endl;
    //    }
    std::cout << "done." << std::endl;
}
