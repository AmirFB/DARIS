cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(fgprs)

set(CUDA_HOME "/usr/local/cuda-11.6")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.6/bin/nvcc")
set(CUDNN_LIBRARY_PATH "/home/amir/anaconda3/envs/pt/lib/libcudnn.so")
set(CUDNN_INCLUDE_PATH "/home/amir/anaconda3/envs/pt/include")

# set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")
# set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")

set(FGPRS_INCLUDE_PATH src)

find_package(Torch REQUIRED)

add_executable(fgprs main.cpp)
add_subdirectory(${FGPRS_INCLUDE_PATH})
target_link_libraries(fgprs ${TORCH_LIBRARIES})
# target_link_libraries(fgprs src/sch)
set_property(TARGET fgprs PROPERTY CXX_STANDARD 14)