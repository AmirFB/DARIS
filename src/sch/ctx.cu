# include <ctx.h>

# include <schd.h>

# include <iostream>

# include <cuda.h>
# include <cudaTypedefs.h>
# include <cuda_runtime.h>

# include <torch/torch.h>

using namespace FGPRS;

using namespace std;
using namespace torch;

MyContext::MyContext()
{
	_default = true;
	this->smCount = Scheduler::maxSmCount;
	queueDuration = 0;
	_pMutex = new mutex();
}

MyContext::MyContext(unsigned smCount)
{
	_default = false;
	this->smCount = smCount;
	queueDuration = 0;
	_pMutex = new mutex();
}

bool MyContext::initialize()
{
	if (_default)
		return true;

	CUexecAffinityParam_v1 affinity;
	affinity.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
	affinity.param.smCount.val = smCount;
	auto result = cuCtxCreate_v3(&_context, &affinity, 1, 0, 0);
	
	return result == CUDA_SUCCESS;
}

bool MyContext::select(double duration)
{
	queueDuration += (unsigned long)(duration * 1000000);

	if (_default)
		return MyContext::selectDefault();
	
	// if (busy)
	// 	return false;

	busy = true;

	return cuCtxSetCurrent(_context) == CUDA_SUCCESS;
}

bool MyContext::selectDefault()
{
	return cuCtxSetCurrent(0) == CUDA_SUCCESS;
}

bool MyContext::release(double duration)
{
	queueDuration -= (unsigned long)(duration * 1000000);
	busy = false;
	// cuCtxSynchronize();
	// torch::cuda::synchronize();
	return selectDefault();
}

bool MyContext::destroy()
{
	selectDefault();
	
	if (_default)
		return true;
	
	return cuCtxDestroy(_context) == CUDA_SUCCESS;
}

void MyContext::lock()
{
	_pMutex->lock();
}

void MyContext::unlock()
{
	_pMutex->unlock();
}