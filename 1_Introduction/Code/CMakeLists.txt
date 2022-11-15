cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project ("Cuda Intro 1" VERSION 1.0.0 LANGUAGES CXX CUDA)

find_package(CUDAToolkit 10.2 REQUIRED)
add_subdirectory(tiny-eigen)

# CMake automatically compiles .cu file with NVCC
add_executable(hello_world hello_world.cu)
target_link_libraries(hello_world CUDA::cudart)

# Compile for Pascal and newer cards
set_property(TARGET hello_world PROPERTY CUDA_ARCHITECTURES 52-virtual)


add_executable(simple_particle simple_particle.cu)
target_link_libraries(simple_particle Eigen3::Eigen CUDA::cudart)
set_property(TARGET simple_particle PROPERTY CUDA_ARCHITECTURES 52-virtual 61-virtual 75-virtual 80-virtual)
