bash build.sh

echo quit | nvidia-cuda-mps-control
nvidia-cuda-mps-control -d
# export CUDA_LAUNCH_BLOCKING=1
# export PYTORCH_NO_CUDA_MEMORY_CACHING=1
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync,garbage_collection_threshold:0.8
# export CUDA_HOME=/usr/local/cuda
# export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
# export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
# export PYTORCH_NO_CUDA_MEMORY_CACHING=1

sudo nvidia-smi -pl 280

# export TORCH_NO_GRAD=1

# ./build/fgprs proposed clear

./build/fgprs mps 30 2000 2
./build/fgprs mps 30 2000 2
./build/fgprs mps 30 2000 2
./build/fgprs mps 30 2000 2
./build/fgprs mps 30 2000 2

for dist in {1..2}
do
	for sms in {3..3}
	do
		for mode in {1..3}
		do
			# ./build/fgprs proposed 30 1000 $mode $dist $sms
			# ./build/fgprs proposed 30 1000 $mode $dist $sms
		done
	done
done

echo quit | nvidia-cuda-mps-control