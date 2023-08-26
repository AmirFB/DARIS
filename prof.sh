bash build.sh

echo quit | nvidia-cuda-mps-control
nvidia-cuda-mps-control -d
# export CUDA_LAUNCH_BLOCKING=0
# export TORCH_CUDA_SANITIZER=1
# export PYTORCH_NO_CUDA_MEMORY_CACHING=1
# export PYTORCH_CUDA_ALLOC_CONF=backend:native,garbage_collection_threshold:0.9

./build/fgprs clear speedup
# ./build/fgprs concurrency 1 1 200

# /usr/local/cuda/bin/nsys profile -w true --trace=cuda,nvtx,osrt,cudnn,cublas,opengl,openacc,openmp,mpi,vulkan -s cpu --stats=true --cudabacktrace=true --cuda-memory-usage=true --capture-range=cudaProfilerApi --gpu-metrics-device=0 --gpu-metrics-set=0 --gpu-metrics-frequency=10 build/fgprs concurrency 1 1 100

# /usr/local/cuda/bin/nsys profile -w true --trace=cuda,nvtx,osrt,cudnn,cublas,opengl,openacc,openmp,mpi,vulkan -s cpu --stats=true --cudabacktrace=true --cuda-memory-usage=true --capture-range=cudaProfilerApi --gpu-metrics-device=0 --gpu-metrics-set=0 --gpu-metrics-frequency=10 build/fgprs concurrency 1 5 1000

# /usr/local/cuda/bin/nsys profile -w true --trace=cuda,nvtx,osrt,cudnn,cublas,opengl,openacc,openmp,mpi,vulkan -s cpu --stats=true --cudabacktrace=true --cuda-memory-usage=true --capture-range=cudaProfilerApi --gpu-metrics-device=0 --gpu-metrics-set=0 --gpu-metrics-frequency=10 build/fgprs concurrency 2 5 1000

# /usr/local/cuda/bin/nsys profile -w true --trace=cuda,nvtx,osrt,cudnn,cublas,opengl,openacc,openmp,mpi,vulkan -s cpu --stats=true --cudabacktrace=true --cuda-memory-usage=true --capture-range=cudaProfilerApi --gpu-metrics-device=0 --gpu-metrics-set=0 --gpu-metrics-frequency=10 build/fgprs concurrency 3 5 1000

# /usr/local/cuda/bin/nsys profile -w true --trace=cuda,nvtx,osrt,cudnn,cublas,opengl,openacc,openmp,mpi,vulkan -s cpu --stats=true --cudabacktrace=true --cuda-memory-usage=true --capture-range=cudaProfilerApi --gpu-metrics-device=0 --gpu-metrics-set=0 --gpu-metrics-frequency=10 build/fgprs concurrency 4 5 1000

/usr/local/cuda/bin/nsys profile -w true --trace=cuda,nvtx,osrt,cudnn,cublas,opengl,openacc,openmp,mpi,vulkan -s cpu --stats=true --cudabacktrace=true --cuda-memory-usage=true --capture-range=cudaProfilerApi --gpu-metrics-device=0 --gpu-metrics-set=0 --gpu-metrics-frequency=20000 build/fgprs proposed 5 0.7 3 40 50 60 68 50 40 50 60

echo quit | nvidia-cuda-mps-control

# --capture-range=cudaProfilerApi