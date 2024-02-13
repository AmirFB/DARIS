bash build.sh

echo quit | nvidia-cuda-mps-control
nvidia-cuda-mps-control -d
export CUDA_LAUNCH_BLOCKING=0
# export PYTORCH_NO_CUDA_MEMORY_CACHING=1

# export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync,garbage_collection_threshold:0.99
# export PYTORCH_CUDA_ALLOC_CONF=backend:native,garbage_collection_threshold:0.5

sudo nvidia-smi -pl 280

export TORCH_NO_GRAD=1

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
export CMAKE_CUDA_COMPILER=/usr/local/cuda/bin

export PATH=/usr/local/cuda/:$PATH
export PATH=/usr/local/cuda/bin:$PATH
export PATH=/usr/local/cuda/extras/CUPTI/lib64:$PATH
# export PATH=/usr/local/cuda/targets/x86_64-linux/lib:$PATH

# /usr/local/cuda/bin/nsys profile -w true --trace=cuda,nvtx,osrt,cudnn,cublas,opengl,openacc,openmp,mpi,vulkan -s cpu --stats=true --cudabacktrace=true --cuda-memory-usage=true --gpu-metrics-device=0 --gpu-metrics-set=0 --gpu-metrics-frequency=20000 --capture-range=cudaProfilerApi build/fgprs 3 1.25 30 100 3 100 10

/usr/local/cuda/bin/nsys profile -w true --trace=cuda,nvtx,osrt,cudnn,cublas,opengl,openacc,openmp,mpi,vulkan -s cpu --stats=true --cudabacktrace=true --cuda-memory-usage=true --gpu-metrics-device=0 --gpu-metrics-set=0 --gpu-metrics-frequency=10 build/fgprs 3 1.25 30 100 3 1000 10

echo quit | nvidia-cuda-mps-control

# --capture-range=cudaProfilerApi