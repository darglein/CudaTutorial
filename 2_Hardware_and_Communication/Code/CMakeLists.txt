cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project ("Cuda Intro 2" VERSION 1.0.0 LANGUAGES CXX CUDA)

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit 10.2 REQUIRED)
find_package(Eigen3 REQUIRED)

list(APPEND CUDA_FLAGS "--expt-relaxed-constexpr")
list(APPEND CUDA_FLAGS "-Xcompiler=-fopenmp")

include_directories(.)

add_executable(collision_detection collision_detection.cu Timer.h)
target_link_libraries(collision_detection CUDA::cudart Eigen3::Eigen OpenMP::OpenMP_CXX)
set_property(TARGET collision_detection PROPERTY CUDA_ARCHITECTURES 52-virtual)
target_compile_options(collision_detection PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>)
