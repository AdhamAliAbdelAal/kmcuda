cmake_minimum_required(VERSION 3.28)
project(knn CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(knn cuda/knn.cu cuda/io.cu cuda/kernel.cu)

set_target_properties(knn PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)