echo quit | nvidia-cuda-mps-control
nvidia-cuda-mps-control -d
export CUDA_LAUNCH_BLOCKING=1
# export PYTORCH_NO_CUDA_MEMORY_CACHING=1
for var in {2..68..2}
do
	./build/fgprs $var
done
echo quit | nvidia-cuda-mps-control