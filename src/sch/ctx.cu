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

MyContext::MyContext(unsigned smCount, int index, bool isDefault)
{
	_default = isDefault;
	this->smCount = smCount;
	queueDuration = 0;
	_pMutex = new mutex();
	this->index = index;
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

bool MyContext::select()
{
	if (_default)
		return MyContext::selectDefault();

	busy = true;

	return cuCtxSetCurrent(_context) == CUDA_SUCCESS;
}

bool MyContext::selectDefault()
{
	return cuCtxSetCurrent(0) == CUDA_SUCCESS;
}

bool MyContext::release()
{
	busy = false;
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