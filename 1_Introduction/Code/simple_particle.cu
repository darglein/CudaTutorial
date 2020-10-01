/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include <iostream>
#include <vector>

#include <thrust/device_vector.h>

#include "Eigen/Core"

using vec3 = Eigen::Vector3f;

struct Particle {
  vec3 position;
  float radius;
  vec3 velocity;
  float invMass;
};

__device__ int GlobalThreadId() { return blockIdx.x * gridDim.x + threadIdx.x; }
int iDivUp(int a, int b) { return (a + b - (1)) / b; }

__global__ static void updateParticles(Particle *particles, int N, float dt) {
  int tid = GlobalThreadId();
  if (tid >= N)
    return;
  Particle &p = particles[tid];
  p.position += p.velocity * dt;
  p.velocity += vec3(0, -9.81, 0) * dt;
}

int main(int argc, char *argv[]) {
  const int N = 100;
  const int k = 3;
  float dt = 0.1;
  std::vector<Particle> particles(N);
  // asdfd

  for (Particle &p : particles) {
    p.position = vec3::Zero();
    p.velocity = vec3::Random();
  }

  thrust::device_vector<Particle> d_particles(particles);
  for (int i = 0; i < k; ++i) {
    const int BLOCK_SIZE = 128;
    updateParticles<<<iDivUp(N, BLOCK_SIZE), BLOCK_SIZE>>>(
        d_particles.data().get(), N, dt);
  }

  updateParticles<<<dim3(1, 1), dim3(4, 4)>>>(d_particles.data().get(), N, dt);

  thrust::copy(d_particles.begin(), d_particles.end(), particles.begin());

  for (Particle &p : particles) {
    std::cout << p.position.transpose() << " " << p.velocity.transpose()
              << std::endl;
  }
  std::cout << "done." << std::endl;
}
