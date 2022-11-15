# Standalone cmakelist for the hello cuda example.
cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project ("Cuda Intro 4" VERSION 1.0.0 LANGUAGES CXX CUDA)


find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)
add_subdirectory(tiny-eigen)

list(APPEND MY_CUDA_FLAGS "--expt-relaxed-constexpr")
list(APPEND MY_CUDA_FLAGS "-Xcompiler=-fopenmp")
list(APPEND MY_CUDA_FLAGS "-Xcompiler=-g")
list(APPEND MY_CUDA_FLAGS "-lineinfo")



add_executable(collision_detection collision_detection.cu)
target_link_libraries(collision_detection Eigen3::Eigen CUDA::cudart OpenMP::OpenMP_CXX)
set_property(TARGET collision_detection PROPERTY CUDA_ARCHITECTURES 52-virtual 75)
target_compile_options(collision_detection PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${MY_CUDA_FLAGS}>)


add_executable(simple_particle simple_particle.cu)
target_link_libraries(simple_particle Eigen3::Eigen CUDA::cudart OpenMP::OpenMP_CXX)
set_property(TARGET simple_particle PROPERTY CUDA_ARCHITECTURES 52-virtual 75)
target_compile_options(simple_particle PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${MY_CUDA_FLAGS}>)

add_executable(streams streams.cu)
target_link_libraries(streams Eigen3::Eigen CUDA::cudart OpenMP::OpenMP_CXX)
set_property(TARGET streams PROPERTY CUDA_ARCHITECTURES 52-virtual 75)
target_compile_options(streams PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${MY_CUDA_FLAGS}>)
