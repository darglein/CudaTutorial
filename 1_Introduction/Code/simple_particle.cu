/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "tiny-eigen/matrix.h"

#include <iostream>
#include <vector>

#include <thrust/device_vector.h>

using vec3 = Eigen::Matrix<float, 3, 1>;

struct Particle
{
    vec3 position;
    float radius;
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

int main(int argc, char* argv[])
{
    const int N     = 25;
    const int steps = 3;
    float dt        = 0.1;

    // Allocate CPU and GPU memory
    std::vector<Particle> particles(N);
    thrust::device_vector<Particle> d_particles(N);

    auto rand_float = []() { return ((rand() % 10000) / 10000.f) * 2 - 1; };

    // Initialize on the CPU
    for (Particle& p : particles)
    {
        p.position = vec3::Zero();
        p.velocity = vec3(rand_float(), rand_float(), rand_float());
    }

    // Upload memory
    thrust::copy(particles.begin(), particles.end(), d_particles.begin());

    // Integrate
    for (int i = 0; i < steps; ++i)
    {
        const int BLOCK_SIZE = 128;
        updateParticles<<<iDivUp(N, BLOCK_SIZE), BLOCK_SIZE>>>(
            d_particles.data().get(), N, dt);
    }

    // Download
    thrust::copy(d_particles.begin(), d_particles.end(), particles.begin());

    for (Particle& p : particles)
    {
        std::cout << p.position.transpose() << " " << p.velocity.transpose()
                  << std::endl;
    }
    std::cout << "done." << std::endl;
}
