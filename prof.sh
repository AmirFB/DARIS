bash build.sh

echo quit | nvidia-cuda-mps-control
nvidia-cuda-mps-control -d
export CUDA_LAUNCH_BLOCKING=1
# export TORCH_CUDA_SANITIZER=1
/usr/local/cuda/bin/nsys profile -w true --trace=cuda,nvtx,osrt,cudnn,cublas,opengl,openacc,openmp,mpi,vulkan -s cpu --stats=true --cudabacktrace=true --cuda-memory-usage=true --capture-range=cudaProfilerApi --gpu-metrics-device=0 --gpu-metrics-set=0 --gpu-metrics-frequency=10 build/fgprs mps 10 0.7 3 40 50 60 68 50 40 50 60

echo quit | nvidia-cuda-mps-control

# --capture-range=cudaProfilerApi