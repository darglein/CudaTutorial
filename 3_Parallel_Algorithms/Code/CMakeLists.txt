cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project ("Cuda Intro 3" VERSION 1.0.0 LANGUAGES CXX CUDA)


find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED)

list(APPEND CUDA_FLAGS "--expt-relaxed-constexpr")

include_directories(.)
FILE(GLOB_RECURSE ALL_HEADER  "*.h")

add_executable(reduce reduce.cu ${ALL_HEADER})
target_link_libraries(reduce CUDA::cudart Eigen3::Eigen)
set_property(TARGET reduce PROPERTY CUDA_ARCHITECTURES 52-virtual)

add_executable(reduce_no_atomic reduce_no_atomic.cu ${ALL_HEADER})
target_link_libraries(reduce_no_atomic CUDA::cudart Eigen3::Eigen)
set_property(TARGET reduce_no_atomic PROPERTY CUDA_ARCHITECTURES 52-virtual)


add_executable(bitonic_sort bitonic_sort.cu ${ALL_HEADER})
target_link_libraries(bitonic_sort CUDA::cudart Eigen3::Eigen)
set_property(TARGET bitonic_sort PROPERTY CUDA_ARCHITECTURES 52-virtual)


add_executable(radix_sort radix_sort.cu ${ALL_HEADER})
target_link_libraries(radix_sort CUDA::cudart Eigen3::Eigen)
set_property(TARGET radix_sort PROPERTY CUDA_ARCHITECTURES 52-virtual)
