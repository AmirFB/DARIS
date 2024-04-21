# include <ctx.hpp>

# include <schd.hpp>

# include <iostream>
# include <random>

# include <cuda.h>
# include <cudaTypedefs.h>
# include <cuda_runtime.h>

# include <torch/torch.h>
# include <c10/cuda/CUDAStream.h>

using namespace FGPRS;

using namespace std;
using namespace torch;

int MyContext::mainStreamCount;
int MyContext::secondaryStreamCount;

MyContext::MyContext(int index, int smCount, bool isDefault) :
	index(index), smCount(smCount), _default(isDefault),
	_pMutex(new mutex), _pQueueMutex(new mutex),
	cv(new condition_variable())
{
	highLastDelayed = vector<shared_ptr<Operation>>(0);
	highLast = vector<shared_ptr<Operation>>(0);
	highDelayed = vector<shared_ptr<Operation>>(0);
	highOther = vector<shared_ptr<Operation>>(0);
	lowLastDelayed = vector<shared_ptr<Operation>>(0);
	lowLast = vector<shared_ptr<Operation>>(0);
	lowDelayed = vector<shared_ptr<Operation>>(0);
	lowOther = vector<shared_ptr<Operation>>(0);
	running = vector<shared_ptr<Operation>>(0);

	remainingSecondaryStreams = secondaryStreamCount;
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
		result &= cuCtxSetCurrent(_context) == CUDA_SUCCESS;
		result &= cuInit(0) == CUDA_SUCCESS;
	}

	size_t temp;

	result &= cuCtxGetLimit(&temp, CU_LIMIT_PRINTF_FIFO_SIZE) == CUDA_SUCCESS || _default;
	result &= cuCtxSetLimit(CU_LIMIT_PRINTF_FIFO_SIZE, temp + (index << 8)) == CUDA_SUCCESS || _default;

	result &= cuCtxGetLimit(&temp, CU_LIMIT_MALLOC_HEAP_SIZE) == CUDA_SUCCESS || _default;
	result &= cuCtxSetLimit(CU_LIMIT_MALLOC_HEAP_SIZE,
		temp + ((Scheduler::contextCount + 1) << 16)) == CUDA_SUCCESS || _default;

	for (int i = 0; i < mainStreamCount; i++)
		_mainStreams.push_back(MyStream(this));

	for (int i = 0; i < secondaryStreamCount; i++)
		_secondaryStreams.push_back(MyStream(this));

	return result;
}

bool MyContext::select()
{
	c10::cuda::CUDAStream::setContextIndex(index);
	return cuCtxSetCurrent(_context) == CUDA_SUCCESS;
}

bool MyContext::selectDefault()
{
	return Scheduler::selectDefaultContext()->select();
}

void MyContext::queueOperation(shared_ptr<Operation> operation)
{
	if (operation->highPriority)
	{
		if (!operation->isLast && operation->priorDelayed)
		{
			highLastDelayed.push_back(operation);
			sort(highLastDelayed.begin(), highLastDelayed.end(),
				[](const std::shared_ptr<FGPRS::Operation>& op1, const std::shared_ptr<FGPRS::Operation>& op2)
				{
					return Operation::EDF(op1, op2);
				});
		}

		else if (operation->isLast)
		{
			highLast.push_back(operation);
			sort(highLast.begin(), highLast.end(),
				[](const std::shared_ptr<FGPRS::Operation>& op1, const std::shared_ptr<FGPRS::Operation>& op2)
				{
					return Operation::EDF(op1, op2);
				});
		}

		else if (operation->priorDelayed)
		{
			highDelayed.push_back(operation);
			sort(highDelayed.begin(), highDelayed.end(),
				[](const std::shared_ptr<FGPRS::Operation>& op1, const std::shared_ptr<FGPRS::Operation>& op2)
				{
					return Operation::EDF(op1, op2);
				});
		}

		else
		{
			highOther.push_back(operation);
			sort(highOther.begin(), highOther.end(),
				[](const std::shared_ptr<FGPRS::Operation>& op1, const std::shared_ptr<FGPRS::Operation>& op2)
				{
					return Operation::EDF(op1, op2);
				});
		}
	}

	else
	{
		if (operation->isLast && operation->priorDelayed)
		{
			lowLastDelayed.push_back(operation);
			sort(lowLastDelayed.begin(), lowLastDelayed.end(),
				[](const std::shared_ptr<FGPRS::Operation>& op1, const std::shared_ptr<FGPRS::Operation>& op2)
				{
					return Operation::EDF(op1, op2);
				});
		}

		else if (operation->isLast)
		{
			lowLast.push_back(operation);
			sort(lowLast.begin(), lowLast.end(),
				[](const std::shared_ptr<FGPRS::Operation>& op1, const std::shared_ptr<FGPRS::Operation>& op2)
				{
					return Operation::EDF(op1, op2);
				});
		}

		else if (operation->priorDelayed)
		{
			lowDelayed.push_back(operation);
			sort(lowDelayed.begin(), lowDelayed.end(),
				[](const std::shared_ptr<FGPRS::Operation>& op1, const std::shared_ptr<FGPRS::Operation>& op2)
				{
					return Operation::EDF(op1, op2);
				});
		}

		else
		{
			lowOther.push_back(operation);
			sort(lowOther.begin(), lowOther.end(),
				[](const std::shared_ptr<FGPRS::Operation>& op1, const std::shared_ptr<FGPRS::Operation>& op2)
				{
					return Operation::EDF(op1, op2);
				});
		}
	}
}

void MyContext::dequeueOperation(shared_ptr<Operation> operation)
{
	if (operation->highPriority)
	{
		if (operation->isLast && operation->priorDelayed)
			highLastDelayed.erase(remove(highLastDelayed.begin(), highLastDelayed.end(), operation), highLastDelayed.end());

		else if (operation->isLast)
			highLast.erase(remove(highLast.begin(), highLast.end(), operation), highLast.end());

		else if (operation->priorDelayed)
			highDelayed.erase(remove(highDelayed.begin(), highDelayed.end(), operation), highDelayed.end());

		else
			highOther.erase(remove(highOther.begin(), highOther.end(), operation), highOther.end());
	}

	else
	{
		if (operation->isLast && operation->priorDelayed)
			lowLastDelayed.erase(remove(lowLastDelayed.begin(), lowLastDelayed.end(), operation), lowLastDelayed.end());

		else if (operation->isLast)
			lowLast.erase(remove(lowLast.begin(), lowLast.end(), operation), lowLast.end());

		else if (operation->priorDelayed)
			lowDelayed.erase(remove(lowDelayed.begin(), lowDelayed.end(), operation), lowDelayed.end());

		else
			lowOther.erase(remove(lowOther.begin(), lowOther.end(), operation), lowOther.end());
	}
}

void MyContext::releaseOperation(shared_ptr<Operation> operation)
{
	_pQueueMutex->lock();
	operation->released = false;

	// selectHeadOperation();

	if (_runningCount < mainStreamCount && _headOperation == nullptr)
	{
		_runningCount++;
		running.push_back(operation);
		_pQueueMutex->unlock();
		return;
	}

	queueOperation(operation);
	selectHeadOperation();
	_pQueueMutex->unlock();
	unique_lock<mutex> lock(*_pMutex);

	while (_runningCount >= mainStreamCount || operation != _headOperation)
		cv->wait(lock);

	_pQueueMutex->lock();
	_runningCount++;
	operation->released = true;

	running.push_back(operation);
	dequeueOperation(operation);
	_pQueueMutex->unlock();
}

void MyContext::selectHeadOperation()
{
	shared_ptr<Operation> potentialHead = nullptr;

	if (highLastDelayed.size() > 0)
		potentialHead = highLastDelayed[0];

	else if (highLast.size() > 0)
		potentialHead = highLast[0];

	else	if (highDelayed.size() > 0)
		potentialHead = highDelayed[0];

	else if (highOther.size() > 0)
		potentialHead = highOther[0];

	else if (lowLastDelayed.size() > 0)
		potentialHead = lowLastDelayed[0];

	else if (lowDelayed.size() > 0)
		potentialHead = lowDelayed[0];

	else if (lowLast.size() > 0)
		potentialHead = lowLast[0];

	else if (lowOther.size() > 0)
		potentialHead = lowOther[0];

	_headOperation = potentialHead;
}

void MyContext::finishOperation(shared_ptr<Operation> operation)
{
	_pQueueMutex->lock();

	_runningCount--;
	running.erase(remove(running.begin(), running.end(), operation), running.end());

	selectHeadOperation();

	cv->notify_all();
	_pQueueMutex->unlock();
}

void MyContext::updateUtilization()
{
	double hUtil = 0, aUtil = 0, oUtil = 0;

	for (auto mod : highContainers)
		hUtil += mod->utilizationPartitioned;

	oUtil = hUtil;
	aUtil = hUtil;

	for (auto mod : lowContainers)
	{
		oUtil += mod->utilizationPartitioned;
		aUtil += mod->active ? mod->utilizationPartitioned : 0;
	}

	highUtilization = hUtil / mainStreamCount;
	activeUtilization = aUtil / mainStreamCount;
	overallUtilization = oUtil / mainStreamCount;
}

void MyContext::assignModule(shared_ptr<MyContainer> container)
{
	if (container->highPriority)
		highContainers.push_back(container);
	else
		lowContainers.push_back(container);

	allContainers.push_back(container);
	container->currentContext = this;

	updateUtilization();
}

void MyContext::removeModule(shared_ptr<MyContainer> container)
{
	if (container->highPriority)
		highContainers.erase(remove(highContainers.begin(), highContainers.end(), container), highContainers.end());
	else
		lowContainers.erase(remove(lowContainers.begin(), lowContainers.end(), container), lowContainers.end());

	allContainers.erase(remove(allContainers.begin(), allContainers.end(), container), allContainers.end());

	updateUtilization();
}

void MyContext::warmup()
{
	int strIndex = 0;
	select();

	for (auto container : allContainers)
	{
		for (auto str : _mainStreams)
		{
			str.select();
			container->forwardRandom();
			str.release();
		}

		for (auto str : _secondaryStreams)
		{
			str.select();
			container->forwardRandom();
			str.release();
		}
	}
}

void MyContext::runDummies(shared_ptr<MyContainer> module)
{
	_stopDummies = false;

	_dummyThread = thread([this, module]()
		{
			int maxCount = mainStreamCount - (module->currentContext == this);

			if (maxCount == 0)
				return;

			while (!_stopDummies)
			{
				thread* th = new thread[maxCount];

				for (int i = 0; i < maxCount; i++)
				{
					th[i] = thread([this, module]()
						{
							while (!_stopDummies)
							{
								shared_ptr<MyContainer> cnt;

								while (true)
								{
									cnt = allContainers[rand() % allContainers.size()];

									if (cnt == module || cnt->isDummy)
										continue;

									break;
								}

								cnt->runDummy();
							}
						});
				}

				for (int i = 0; i < maxCount; i++)
					th[i].join();
			}
		});
}

void MyContext::stopDummies()
{
	_stopDummies = true;
}

void MyContext::waitDummies()
{
	_dummyThread.join();
}

MyStream* MyContext::getStream()
{
	for (auto& stream : _mainStreams)
		if (!stream.busy)
			return &stream;

	return nullptr;
}

MyStream* MyContext::getSecondaryStream(MyStream* mainStream)
{
	for (auto& stream : _secondaryStreams)
		if (!stream.busy)
			return &stream;

	return mainStream;
}