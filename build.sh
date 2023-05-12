clear
mkdir build
cd build
rm fgprs

export CUDA_HOME=/usr/local/cuda
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CFLAGS="-I$CUDA_HOME/include $CFLAGS"
# export $CMAKE_CUDA_COMPILER=/home/amir/anaconda3/envs/lt-source/bin/nvcc

# export LDFLAGS="-libc++=libstdc++"
# export CXXFLAGS="-lstdc++fs -std=c++20"
export TORCH_USE_CUDA_DSA=ON
cmake -DCMAKE_PREFIX_PATH=/home/amir/repos/FGPRS/libtorch-install/ -DTORCH_USE_CUDA_DSA=ON ..
cmake --build . --config Release -j32