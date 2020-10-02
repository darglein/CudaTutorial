# Standalone cmakelist for the hello cuda example.
cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project ("Cuda Intro 3" VERSION 1.0.0 LANGUAGES CXX CUDA)


find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED)

FILE(GLOB_RECURSE ALL_HEADER  "*.h")


SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -fopenmp")

include_directories(.)

add_executable(reduce reduce.cu ${ALL_HEADER})
target_link_libraries(reduce CUDA::cudart Eigen3::Eigen)
set_property(TARGET reduce PROPERTY CUDA_ARCHITECTURES 50)


add_executable(bitonic_sort bitonic_sort.cu ${ALL_HEADER})
target_link_libraries(bitonic_sort CUDA::cudart Eigen3::Eigen)
set_property(TARGET bitonic_sort PROPERTY CUDA_ARCHITECTURES 50)


add_executable(radix_sort radix_sort.cu ${ALL_HEADER})
target_link_libraries(radix_sort CUDA::cudart Eigen3::Eigen)
set_property(TARGET radix_sort PROPERTY CUDA_ARCHITECTURES 50)
