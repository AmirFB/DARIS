cmake -DUSE_CUDA=1 -DUSE_CUDA_PATH=/usr/local/cuda -DUSE_CUDNN=1 -DUSE_MKLDNN=1 -DUSE_CPP_PACKAGE=1 -GNinja ..
ninja -v