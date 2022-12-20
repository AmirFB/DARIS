mkdir libtorch-build
cd libtorch-build

export CUDA_HOME=$HOME/usr/local/cuda
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CFLAGS="-I$CUDA_HOME/include $CFLAGS"

# export CC=/usr/bin/gcc-10
# export CXX=/usr/bin/g++-10

cmake -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_BUILD_TYPE:STRING=Release -DPYTHON_EXECUTABLE:PATH=`which python3` -DCMAKE_INSTALL_PREFIX:PATH=../libtorch-install ../../pytorch
# cmake --build . --target clean -j32
cmake --build . --target install -j32

# cmake CMAKE_GENERATOR="-GNinja" CMAKE_INSTALL="ninja install" -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_BUILD_TYPE:STRING=Release -DPYTHON_EXECUTABLE:PATH=`which python3` -DCMAKE_INSTALL_PREFIX:PATH=../libtorch-install ../../pytorch
# ninja install