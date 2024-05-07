# clear
# mkdir build
# cd build
# rm fgprs

# export CUDA_HOME=/usr/local/cuda
# export CUDA_BIN_PATH=/usr/local/cuda/bin
# export CUDA_INC_PATH=/usr/local/cuda/include
# export CUDA_LIB_PATH=/usr/local/cuda/lib
# export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
# export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# export CFLAGS="-I$CUDA_HOME/include $CFLAGS"
# export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64"
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
# export CUDA_INCLUDE_DIRS=/usr/local/cuda/include
# # export CMAKE_CUDA_COMPILER=/usr/local/cuda/bin

# export PATH=/usr/local/cuda/:$PATH
# export PATH=/usr/local/cuda/bin:$PATH

# # export LDFLAGS="-libc++=libstdc++"
# export CXXFLAGS="-lstdc++fs -std=c++20"
# export TORCH_USE_CUDA_DSA=ON

# export CFLAGS="-I/home/amir/repos/FGPRS/libsmctrl $CFLAGS"
# export LDFLAGS="-lsmctrl $LDFLAGS"
# export LDFLAGS="-L/home/amir/repos/FGPRS/libsmctrl -lsmctrl $LDFLAGS"

# echo "Exported variables:"
# echo $CFLAGS
# echo $LDFLAGS

# export CMAKE_PREFIX_PATH=/home/amir/repos/FGPRS/libtorch-install/
# cmake -DTORCH_USE_CUDA_DSA=ON ..
# cmake --build . --config Release -j32






clear
mkdir build
cd build
rm fgprs

export CUDA_HOME=/usr/local/cuda
export CUDA_BIN_PATH=/usr/local/cuda/bin
export CUDA_INC_PATH=/usr/local/cuda/include
export CUDA_LIB_PATH=/usr/local/cuda/lib
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CFLAGS="-I$CUDA_HOME/include $CFLAGS"
export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
export CUDA_INCLUDE_DIRS=/usr/local/cuda/include
# export CMAKE_CUDA_COMPILER=/usr/local/cuda/bin

export PATH=/usr/local/cuda/:$PATH
export PATH=/usr/local/cuda/bin:$PATH

# export LDFLAGS="-libc++=libstdc++"
export CXXFLAGS="-lstdc++fs -std=c++20"
export TORCH_USE_CUDA_DSA=ON

export CFLAGS="-I/home/amir/repos/FGPRS/libsmctrl $CFLAGS"
export LDFLAGS="-lsmctrl $LDFLAGS"
export LDFLAGS="-L/home/amir/repos/FGPRS/libsmctrl -lsmctrl $LDFLAGS"

echo "Exported variables:"
echo $CFLAGS
echo $LDFLAGS

export CMAKE_PREFIX_PATH=/home/amir/repos/FGPRS/libtorch-install/
cmake -DTORCH_USE_CUDA_DSA=ON ..
cmake --build . --config Release -j32