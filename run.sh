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

# ./build/fgprs 2 10
# ./build/fgprs 2 10
# ./build/fgprs 2 10
# ./build/fgprs 2 10
# ./build/fgprs 2 10

./build/fgprs 1 1 3 3 1 500 5

# for dummy in $(seq 1 1 10)
# do
# 	echo "ROUND $dummy"

# 	for ctx in $(seq 1 1 8)
# 	do
# 		min=1

# 		if [ "$ctx" == 1 ]; then
# 			min=2
# 		fi

# 		limit=$(echo "10 / $ctx" | bc)

# 		if [ "$limit" -gt 5 ]; then
# 			limit=10
# 		fi

# 		for str in $(seq $min 1 $limit)
# 		do
# 			if [ "$ctx" == 1 ]; then
# 				os_options=("1")
				
# 			elif [ "$ctx" == 2 ]; then
# 				os_options=("1" "1.5" "2")
			
# 			elif [ "$ctx" -gt 2 ]; then
# 				os_options=("1" "1.5" "2" "$ctx")
# 			fi

# 			for os in ${os_options[@]}
# 			do
# 				for ts in $(seq 3 1 3)
# 				do
# 					# echo "Running with $ts $ctx $str $os 5000 3"
# 					./build/fgprs 1 $ts $ctx $str $os 5000 5
# 				done
# 			done
# 		done
# 	done
# done

echo quit | nvidia-cuda-mps-control