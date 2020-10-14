cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project ("Cuda Intro 1" VERSION 1.0.0 LANGUAGES CXX CUDA)

# CMake automatically compiles .cu file with NVCC
add_executable(hello_world hello_world.cu)

# Compile for Pascal and newer cards
set_property(TARGET hello_world PROPERTY CUDA_ARCHITECTURES 52-virtual)


find_package(Eigen3 REQUIRED)


#list(APPEND MY_CUDA_FLAGS "-use_fast_math")
list(APPEND MY_CUDA_FLAGS "--expt-relaxed-constexpr")
list(APPEND MY_CUDA_FLAGS "-Xcompiler=-fopenmp")
list(APPEND MY_CUDA_FLAGS "-G")

target_compile_options(hello_world PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${MY_CUDA_FLAGS}>)

message("${TESTA}")
add_executable(simple_particle simple_particle.cu)
target_link_libraries(simple_particle Eigen3::Eigen)
set_property(TARGET simple_particle PROPERTY CUDA_ARCHITECTURES 52-virtual 61-virtual 75-virtual 80-virtual)
