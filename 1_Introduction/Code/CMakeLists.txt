# Standalone cmakelist for the hello cuda example.
cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project ("Cuda Intro 1" VERSION 1.0.0 LANGUAGES CXX CUDA)


find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(.)

add_executable(hello_world hello_world.cu)
target_link_libraries(hello_world CUDA::cudart Eigen3::Eigen)
set_property(TARGET hello_world PROPERTY CUDA_ARCHITECTURES 50)


add_executable(simple_particle simple_particle.cu)
target_link_libraries(simple_particle CUDA::cudart Eigen3::Eigen)
set_property(TARGET simple_particle PROPERTY CUDA_ARCHITECTURES 50)
