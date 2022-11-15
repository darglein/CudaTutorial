/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "tiny-eigen/matrix.h"
#include "Timer.h"

#include <iostream>
#include <vector>

#include <cuda_profiler_api.h>
#include <thrust/device_vector.h>

using vec3 = Eigen::Matrix<float, 3, 1>;

// ===== Helper functions ====
__device__ inline int GlobalThreadId()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
}
inline int iDivUp(int a, int b)
{
    return (a + b - (1)) / b;
}



struct __align__(16) Particle
{
    vec3 position;
    float radius;
    vec3 velocity;
    float invMass;
};

__global__ static void IntegrateParticlesSimple(Particle* particles, int N, float dt)
{
    int tid = GlobalThreadId();
    if (tid >= N) return;
    Particle p = particles[tid];
    p.position += p.velocity * dt;
    p.velocity += vec3(0, -9.81, 0) * dt;
    particles[tid] = p;
}

struct __align__(16) PositionRadius
{
    vec3 position;
    float radius;
};


struct __align__(16) VelocityMass
{
    vec3 velocity;
    float invMass;
};

__global__ static void IntegrateParticlesInverse(PositionRadius* prs, VelocityMass* vms, int N, float dt)
{
    int tid = GlobalThreadId();
    if (tid >= N) return;
    PositionRadius pr;
    VelocityMass vm;

    pr = prs[tid];
    vm = vms[tid];

    pr.position += vm.velocity * dt;
    vm.velocity += vec3(0, -9.81, 0) * dt;

    vms[tid] = vm;
    prs[tid] = pr;
}

__global__ static void IntegrateParticlesInverse16(PositionRadius* prs, VelocityMass* vms, int N, float dt)
{
    int tid = GlobalThreadId();
    if (tid >= N) return;
    PositionRadius pr;
    VelocityMass vm;

    reinterpret_cast<int4*>(&pr)[0] = reinterpret_cast<int4*>(prs)[tid];
    reinterpret_cast<int4*>(&vm)[0] = reinterpret_cast<int4*>(vms)[tid];

    pr.position += vm.velocity * dt;
    vm.velocity += vec3(0, -9.81, 0) * dt;

    reinterpret_cast<int4*>(prs)[tid] = reinterpret_cast<int4*>(&pr)[0];
    reinterpret_cast<int4*>(vms)[tid] = reinterpret_cast<int4*>(&vm)[0];
}

int main(int argc, char* argv[])
{
    const int N     = 2500000;
    const int steps = 11;
    float dt        = 0.1;

    std::cout << "Testing Particle integration N = " << N << std::endl;
    std::cout << std::endl;

    // Allocate CPU and GPU memory
    std::vector<Particle> particles(N);
    thrust::device_vector<Particle> d_particles(particles);

    thrust::device_vector<PositionRadius> d_pr(N);
    thrust::device_vector<VelocityMass> d_vm(N);


    cudaProfilerStart();

    float time_simple = measureObject(steps, [&]() {
        const int BLOCK_SIZE = 128;
        IntegrateParticlesSimple<<<iDivUp(N, BLOCK_SIZE), BLOCK_SIZE>>>(d_particles.data().get(), N, dt);
    });
    std::cout << "IntegrateParticlesSimple    " << time_simple << " ms." << std::endl;

    float time_inverse = measureObject(steps, [&]() {
        const int BLOCK_SIZE = 128;
        IntegrateParticlesInverse<<<iDivUp(N, BLOCK_SIZE), BLOCK_SIZE>>>(d_pr.data().get(), d_vm.data().get(), N, dt);
    });
    std::cout << "IntegrateParticlesInverse   " << time_inverse << " ms." << std::endl;

    float time_inverse16 = measureObject(steps, [&]() {
        const int BLOCK_SIZE = 128;
        IntegrateParticlesInverse16<<<iDivUp(N, BLOCK_SIZE), BLOCK_SIZE>>>(d_pr.data().get(), d_vm.data().get(), N, dt);
    });
    std::cout << "IntegrateParticlesInverse16 " << time_inverse16 << " ms." << std::endl;

    cudaProfilerStop();

    std::cout << "done." << std::endl;
}
