echo quit | nvidia-cuda-mps-control
nvidia-cuda-mps-control -d
export CUDA_LAUNCH_BLOCKING=1
/usr/local/cuda/bin/nsys profile -t nvtx,cuda --stats=true --cudabacktrace memory --cuda-memory-usage=true --gpu-metrics-device=0 --gpu-metrics-set=0 --gpu-metrics-frequency=10000 build/fgprs interference 68 1000
echo quit | nvidia-cuda-mps-control
