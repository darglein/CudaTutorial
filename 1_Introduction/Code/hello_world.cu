/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include <iostream>

#include <cuda_runtime.h>

__global__ void helloCudaKernel() {
  printf("Hello from thread %d!\n", threadIdx.x);
}

int main(int argc, char *argv[]) {
  helloCudaKernel<<<1, 8>>>();
  cudaDeviceSynchronize();

  return 0;
}
