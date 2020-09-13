cmake_minimum_required(VERSION 3.17)
project(CudaMandel CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(CudaMandel main.cu png.cu mandelbrot.cu)

include_directories(lodepng)
include_directories(gpuprec/gqd)


target_link_libraries(CudaMandel lodepng)

set_target_properties(
        CudaMandel
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON CMAKE_CUDA_FLAGS "-fmad=false")