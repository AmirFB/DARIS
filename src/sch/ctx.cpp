# include <ctx.hpp>

# include <ctxd.hpp>
# include <schd.hpp>

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
	_pMutex = new mutex();
	this->index = index;
}

bool MyContext::initialize()
{
	if (_default)
	{
		cuCtxGetCurrent(&_context);
		return true;
	}

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
	return cuCtxSetCurrent(NULL) == CUDA_SUCCESS;
}

bool MyContext::release()
{
	busy = false;
	return selectDefault();
}

void MyContext::lock()
{
	_pMutex->lock();
}

void MyContext::unlock()
{
	_pMutex->unlock();
}

void MyContext::queueOperation(Operation* operation)
{
	_changed = true;
	queue.push_back(operation);
}

void MyContext::dequeueOperation()
{
	_changed = true;
	queue.pop_front();
}

steady_clock::time_point MyContext::getFinishTime()
{
	if (queue.size() == 0)
		return steady_clock::now();

	if (!_changed)
		return _finishTime;

	double sum = 0;

	for (auto op : queue)
		sum += op->contextData[index].occupiedExecutionTime;

	_changed = false;
	_finishTime = queue[0]->startTime + microseconds((int)sum);
	return _finishTime;
}

bool MyContext::isEmpty()
{
	return queue.size() == 0;
}