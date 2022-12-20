echo quit | nvidia-cuda-mps-control
nvidia-cuda-mps-control -d
export CUDA_LAUNCH_BLOCKING=1
export CUDA_HOME=/usr/local/cuda
# export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
# export PYTORCH_NO_CUDA_MEMORY_CACHING=1
for var in {2..2..2}
do
	./build/fgprs $var
done
echo quit | nvidia-cuda-mps-control