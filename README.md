# Introduction to GPU Programming with CUDA

This course covers the basics of GPU programming using the CUDA API.
It consists of 4 lecture-style presentation as well as the sample code. 
The content was created for the class *Advanced Game Physics* at FA-University of Erlangen-Nuremberg.
The target audience are undergraduate students with basic programming knowledge in C/C++.

## Lecture 1: The CUDA Programming Model
        
 * History of GPGPU
 * Compiling and Running CUDA Programs
 * Launching CUDA Kernels
 * Launch Arguments
 * Hello World Example + Code
 * Host/Device Memory Allocation and Transfer
 * Particle Integration Example + Code
 * OpenGL Interop
 * CUDA-Memcheck
 
[[Slides]](https://github.com/darglein/CudaTutorial/tree/master/1_Introduction/Slides) [[Code]](https://github.com/darglein/CudaTutorial/tree/master/1_Introduction/Code) [[Video]]()

## Lecture 2: Hardware Architecture and Parallel Communication

 * GPU Architecture Overview
 * Streaming Multiprocessors
 * Memory Accesses
 * L1/L2 Cache, Constant Cache, Texture Cache
 * Registers
 * Synchronization
 * Communication (Memory, Registers)
 * Atomic Operations
 * Shared Memory
 * Example: Red-Blue Collision Detection
 
## Lecture 3: Parallel Algorithms - Implementation and Usage
 
 * Reduce
 * Scan
 * Scatter
 * Gather
 * Compact
 * Sort

## Lecture 4: Profiling and Optimizing CUDA Kernels
 
 * Compiling, PTX, SASS
 * CUDA Profiling
 * Memory Access Patterns
 * Streams, Events
 * Async Memcpy, Pinned Memory
