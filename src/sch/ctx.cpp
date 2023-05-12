# include <ctx.hpp>

# include <ctxd.hpp>
# include <schd.hpp>
# include <loop.hpp>

# include <iostream>

# include <cuda.h>
# include <cudaTypedefs.h>
# include <cuda_runtime.h>

# include <torch/torch.h>
// # include <c10/cuda/CUDAStream.h>

using namespace FGPRS;

using namespace std;
using namespace torch;

MyContext::MyContext(unsigned smCount, int index, bool isDefault) :
	smCount(smCount), index(index), _default(isDefault),
	_pMutex(new mutex), _pQueueMutex(new mutex), //_lock(*_pMutex),
	cv(new condition_variable())
{
}

bool MyContext::initialize()
{
	bool result = true;

	if (_default)
	{
		result &= cuInit(0) == CUDA_SUCCESS;
		result &= cuCtxGetCurrent(&_context) == CUDA_SUCCESS;
	}

	else
	{
		CUexecAffinityParam_v1 affinity;
		affinity.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
		affinity.param.smCount.val = smCount;
		result &= cuCtxCreate_v3(&_context, &affinity, 1, 0, 0) == CUDA_SUCCESS;
		cuInit(0);
	}

	size_t temp;

	result &= cuCtxGetLimit(&temp, CU_LIMIT_PRINTF_FIFO_SIZE) == CUDA_SUCCESS;
	result &= cuCtxSetLimit(CU_LIMIT_PRINTF_FIFO_SIZE, temp + (index << 8)) == CUDA_SUCCESS;

	result &= cuCtxGetLimit(&temp, CU_LIMIT_MALLOC_HEAP_SIZE) == CUDA_SUCCESS;
	result &= cuCtxSetLimit(CU_LIMIT_MALLOC_HEAP_SIZE,
		temp + ((Scheduler::smOptions.size() + (Scheduler::type != PROPOSED_SCHEDULER)) << 16)) == CUDA_SUCCESS;

	result &= cuCtxGetLimit(&temp, CU_LIMIT_MALLOC_HEAP_SIZE) == CUDA_SUCCESS;

	return result;
}

// c10::cuda::CUDAStream* MyContext::select()
bool MyContext::select()
{
	// if (_default)
	// 	return MyContext::selectDefault();

	// busy = true;

	return cuCtxSetCurrent(_context) == CUDA_SUCCESS;
}

bool MyContext::selectDefault()
{
	return Scheduler::selectDefaultContext()->select();
}

bool MyContext::release()
{
	// busy = false;
	// return selectDefault();
}

// void MyContext::lock()
// {
// 	cout << "Locking " << smCount << "\tcount: " << lockCount << endl;
// 	while (lockCount >= 2)
// 	{
// 		cv->wait(_lock);
// 		cout << "Notified " << smCount << "\tcount: " << lockCount << endl;
// 	}
// 	cout << "Locked " << smCount << endl;
// 	lockCount++;
// }

// void MyContext::unlock()
// {
// 	lockCount--;
// 	cv->notify_all();
// 	cout << "Unlocked " << smCount << endl;
// }

void MyContext::lock()
{
	// cout << "Locking " << smCount << "\tcount: " << lockCount << endl;
	unique_lock<mutex> lock(*_pMutex);
	while (lockCount >= 2)
	{
		cv->wait(lock);
		cout << "Notified " << smCount << "\tcount: " << lockCount << endl;
	}
	// cout << "Locked " << smCount << endl;
	lockCount++;
}

void MyContext::unlock()
{
	unique_lock<mutex> lock(*_pMutex);
	lockCount--;
	cout << "Unlocking " << smCount << "\tcount: " << lockCount << endl;
	cv->notify_one();
	cout << "Unlocked " << smCount << endl;
}

void MyContext::queueOperation(Operation* operation)
{
	_pQueueMutex->lock();
	_changed = true;
	queue.push_back(operation);
	_pQueueMutex->unlock();
}

void MyContext::dequeueOperation(Operation* operation)
{
	_pQueueMutex->lock();
	_changed = true;
	queue.erase(std::remove(queue.begin(), queue.end(), operation), queue.end());
	_pQueueMutex->unlock();
}

steady_clock::time_point MyContext::getFinishTime()
{
	if (queue.size() == 0)
		return _finishTime = steady_clock::now();

	if (!_changed)
		return _finishTime;

	_changed = false;
	double sum = 0;

	for (auto op : queue)
		sum += op->contextData[index].occupiedExecutionTime;

	_finishTime = queue[0]->startTime + microseconds((int)sum);
	return _finishTime;
}

bool MyContext::isEmpty()
{
	return queue.size() == 0;
}