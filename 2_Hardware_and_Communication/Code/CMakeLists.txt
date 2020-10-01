# Standalone cmakelist for the hello cuda example.
cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project ("Cuda Intro 2" VERSION 1.0.0 LANGUAGES CXX CUDA)


find_package(OpenMP REQUIRED)
#PackageHelperTarget(OpenMP::OpenMP_CXX OPENMP_FOUND)

find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED)

SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -fopenmp")

include_directories(.)

add_executable(collision_detection collision_detection.cu Timer.h)
target_link_libraries(collision_detection CUDA::cudart Eigen3::Eigen OpenMP::OpenMP_CXX)
set_property(TARGET collision_detection PROPERTY CUDA_ARCHITECTURES 50)

