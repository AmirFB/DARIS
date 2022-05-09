# include <ctx.h>

# include <cuda.h>
# include <cudaTypedefs.h>
# include <cuda_runtime.h>

using namespace FGPRS;

MyContext::MyContext(unsigned smCount)
{
	this->smCount = smCount;
	busy = false;
}

bool MyContext::initialize()
{
	CUexecAffinityParam_v1 affinity;
	_affinity.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
	affinity.param.smCount.val = smCount;

	return cuCtxCreate_v3(&_context, &affinity, 1, 0, 0) == CUDA_SUCCESS;
}

bool MyContext::select()
{
	if (busy)
		return false;
	
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
	cuCtxSynchronize();
	return selectDefault();
}

bool MyContext::destroy()
{
	selectDefault();
	return cuCtxDestroy(_context) == CUDA_SUCCESS;
}