#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/CallOnce.h>

#include <ATen/cuda/CUDAConfig.h>
#include <mutex>
#include <deque>
#include <vector>

# include <iostream>

namespace at { namespace cuda {

namespace {

DeviceIndex num_gpus = -1;
c10::once_flag init_flag;
std::deque<c10::once_flag> device_flags;
std::vector<cudaDeviceProp> device_properties;

void initCUDAContextVectors() {
	// num_gpus = c10::cuda::device_count();
	size_t temp = 0;
	cuCtxGetLimit(&temp, CU_LIMIT_MALLOC_HEAP_SIZE);
	auto count = (temp % 0X800000) >> 16;
	count = count == 0 ? 8 : count;
	num_gpus = count;
	// std::cout << "num_gpus: " << (int)num_gpus << std::endl;
	device_flags.resize(num_gpus);
  device_properties.resize(num_gpus);
}

void initDeviceProperty(DeviceIndex device_index) {
  cudaDeviceProp device_prop;
	AT_CUDA_CHECK(cudaGetDeviceProperties(&device_prop, 0));
	
	CUexecAffinityParam affinity;
	affinity.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
	CUresult result = cuCtxGetExecAffinity(&affinity, CU_EXEC_AFFINITY_TYPE_SM_COUNT);
	int smCount = affinity.param.smCount.val;

	device_prop.multiProcessorCount = smCount;
	device_properties[device_index] = device_prop;
}

} // anonymous namespace

// We need this function to force the linking against torch_cuda(_cpp) on Windows.
// If you need to modify this function, please specify a new function and apply
// the changes according to https://github.com/pytorch/pytorch/pull/34288.
// Related issue: https://github.com/pytorch/pytorch/issues/31611.
/* Device info */
int warp_size() {
  return getCurrentDeviceProperties()->warpSize;
}

cudaDeviceProp* getCurrentDeviceProperties() {
	// auto device = c10::cuda::current_device();
	size_t temp = 0;
	cuCtxGetLimit(&temp, CU_LIMIT_PRINTF_FIFO_SIZE);
	auto device = (temp % 0X1000) >> 8;
	// std::cout << "device: " << device << std::endl;
	return getDeviceProperties(device);
}

bool initial = true;
cudaDeviceProp* getDeviceProperties(int64_t device)
{
  c10::call_once(init_flag, initCUDAContextVectors);
  if (device == -1) device = c10::cuda::current_device();
	AT_ASSERT(device >= 0 && device < num_gpus);
	
	if (initial)
	{
		initDeviceProperty(device);
		if (device != 0)
			initial = false;
	}
	
	else
		c10::call_once(device_flags[device], initDeviceProperty, device);
	
	return &device_properties[device];
}

bool canDeviceAccessPeer(int64_t device, int64_t peer_device) {
  c10::call_once(init_flag, initCUDAContextVectors);
  if (device == -1) device = c10::cuda::current_device();
  AT_ASSERT(device >= 0 && device < num_gpus);
  AT_ASSERT(peer_device >= 0 && peer_device < num_gpus);
  int can_access = 0;
  AT_CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, device, peer_device));
  return can_access != 0;
}

Allocator* getCUDADeviceAllocator() {
  return c10::cuda::CUDACachingAllocator::get();
}

} // namespace cuda

} // namespace at
