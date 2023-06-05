# include <schd.hpp>

# include <iostream>
# include <thread>
# include <future>
# include <ranges>
# include <vector>
# include <unistd.h>
# include <mutex>

# include <cuda.h>
# include <cudaTypedefs.h>
# include <cuda_runtime.h>

# include <torch/torch.h>

using namespace FGPRS;

using namespace std;

using namespace torch;
using namespace torch::nn;

int Scheduler::maxSmCount;
bool Scheduler::_stopDummy;
vector<int> Scheduler::smOptions;
MyContext* Scheduler::_contextPool;
MyContext* Scheduler::_defaultContext;
SchedulerType Scheduler::type;

Sequential* Scheduler::_dummyModule;
Tensor* Scheduler::_dummyInput;

int Scheduler::contextCount = 0;
bool Scheduler::noDefault;

bool Scheduler::initialize(int options[], int size, SchedulerType type, bool noDefault)
{
	bool result = true;
	cudaDeviceProp prop;
	Scheduler::type = type;

	cudaGetDeviceProperties(&prop, 0);
	maxSmCount = prop.multiProcessorCount;
	Scheduler::noDefault = noDefault;
	int dummyCount = size - (noDefault ? 1 : 0);
	contextCount = size + (noDefault ? 0 : 1);

	if (type == PROPOSED_SCHEDULER)
	{
		_dummyModule = new Sequential[dummyCount];
		_dummyInput = new Tensor[dummyCount];

		for (int i = 0; i < dummyCount; i++)
		{
			_dummyInput[i] = torch::randn({ 1, 16, 448, 448 }, kCUDA);
			_dummyModule[i] = Sequential(
				Conv2d(Conv2dOptions(16, 32, 3).stride(1).padding(3)),
				BatchNorm2d(32),
				ReLU(),
				MaxPool2d(MaxPool2dOptions(2))
			);

			_dummyModule[i]->eval();
			_dummyModule[i]->to(kCUDA);
		}

		_contextPool = new MyContext[size + 1];
		smOptions = vector<int>(size + 1);

		smOptions[size] = maxSmCount;
		_contextPool[size] = MyContext(maxSmCount, size, true);
		_defaultContext = &_contextPool[size];
		result &= _contextPool[size].initialize();

		if (!noDefault)
			auto dummy = c10::cuda::getStreamFromPool(false, _contextPool[size].index);
	}

	else
	{
		smOptions = vector<int>(size);
		_contextPool = new MyContext[size];
	}


	for (int i = 0; i < size; i++)
	{
		smOptions[i] = (min(max(options[i], 1), maxSmCount));
		_contextPool[i] = MyContext(options[i], i);
		result &= _contextPool[i].initialize();

		if (type == PROPOSED_SCHEDULER)
			auto dummy = c10::cuda::getStreamFromPool(false, _contextPool[i].index);
	}

	// smOptions.pop_back();
	return result;
}

MyContext* Scheduler::selectContext(int smCount)
{
	for (int i = 0; i < smOptions.size(); i++)
		if (_contextPool[i].smCount >= smCount)
			return &_contextPool[i];

	return _defaultContext;
}

MyContext* Scheduler::selectContextByIndex(int index)
{
	return &_contextPool[index];
}

MyContext* Scheduler::selectDefaultContext()
{
	return &_contextPool[smOptions.size() - 1];
}

bool Scheduler::releaseContext(MyContext context)
{
	return context.release();
}

float Scheduler::getTotalMemoryMB()
{
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	return total / 1024. / 1024.;
}

float Scheduler::getTotalMemoryGB()
{
	return Scheduler::getTotalMemoryMB() / 1024;
}

float Scheduler::getFreeMemoryMB()
{
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	return free / 1024. / 1024.;
}

float Scheduler::getFreeMemoryGB()
{
	return Scheduler::getFreeMemoryMB() / 1024;
}

float Scheduler::getMemoryPercentage()
{
	return Scheduler::getFreeMemoryMB() / Scheduler::getTotalMemoryMB() * 100;
}

void Scheduler::dummyFunction(MyContext* ctx, Sequential* mod, Tensor* in)
{
	NoGradGuard no_grad;
	int counter = 0;
	ctx->select();

	while (!_stopDummy)
	{
		auto dummy = (*mod)->forward(*in);
		counter++;
		cuCtxSynchronize();
	}

	ctx->release();
}

future<void>* Scheduler::_th;

void Scheduler::startDummy(MyContext* ctx)
{
	int index = 0;
	_stopDummy = false;

	_th = new future<void>[contextCount - 1];

	for (int i = 0; i < contextCount; i++)
	{
		if (_contextPool[i].index == ctx->index)
			continue;

		MyContext::selectDefault();
		auto dummy = _dummyModule[index]->forward(_dummyInput[index]);

		_contextPool[i].select();
		dummy = _dummyModule[index]->forward(_dummyInput[index]);
		_contextPool[i].release();

		_th[index] = async(launch::async, dummyFunction, &_contextPool[i], &_dummyModule[index], &_dummyInput[index]);
		index++;
	}
}

void Scheduler::stopDummy()
{
	_stopDummy = true;

	for (int i = 0; i < (contextCount - 1); i++)
		_th[i].get();
}

mutex globalMutex;

MyContext* Scheduler::getMinimalContext(Operation* operation)
{
	MyContext* ctx1 = NULL, * ctx2;
	// globalMutex.lock();

	steady_clock::time_point earliest1 = steady_clock::now() + seconds(1), earliest2 = steady_clock::now() + seconds(1);
	steady_clock::time_point temp;

	for (int i = 0; i < contextCount; i++)
	{
		if (!_contextPool[i].isEmpty())
			continue;

		temp = _contextPool[i].getFinishTime() + microseconds((int)operation->contextData[i].occupiedExecutionTime);

		if (temp < operation->absoluteDeadline)
		{
			_contextPool[i].queueOperation(operation);
			operation->finishTime = _contextPool[i].getFinishTime();
			// globalMutex.unlock();
			operation->queueCount = _contextPool[i].queue.size();
			_contextPool[i].lock(operation);

			return &_contextPool[i];
		}
	}

	for (int i = 0; i < contextCount; i++)
	{
		temp = _contextPool[i].getFinishTime() + microseconds((int)operation->contextData[i].occupiedExecutionTime);

		if (_contextPool[i].isEmpty() && temp < earliest1)
		{
			earliest1 = temp;
			ctx1 = &_contextPool[i];
		}

		if (temp < earliest2)
		{
			earliest2 = temp;
			ctx2 = &_contextPool[i];
		}
	}

	if (ctx1 != NULL)
	{
		ctx1->queueOperation(operation);
		operation->finishTime = ctx1->getFinishTime();
		// globalMutex.unlock();
		operation->queueCount = ctx1->queue.size();
		ctx1->lock(operation);

		return ctx1;
	}


	else
	{
		ctx2->queueOperation(operation);
		operation->finishTime = ctx2->getFinishTime();
		// globalMutex.unlock();
		operation->queueCount = ctx2->queue.size();
		ctx2->lock(operation);

		return ctx2;
	}
}

MyContext* Scheduler::getFastestContext(Operation* operation)
{
	MyContext* ctx;
	// globalMutex.lock();

	steady_clock::time_point earliest = steady_clock::now() + seconds(1);
	steady_clock::time_point temp;

	for (int i = 0; i < contextCount; i++)
	{
		temp = _contextPool[i].getFinishTime() + microseconds((int)operation->contextData[i].occupiedExecutionTime);

		if (temp < earliest)
		{
			earliest = temp;
			ctx = &_contextPool[i];
		}
	}

	ctx->queueOperation(operation);
	operation->finishTime = ctx->getFinishTime();
	// globalMutex.unlock();
	operation->queueCount = ctx->queue.size();
	ctx->lock(operation);

	return ctx;
}

bool Scheduler::anyEmptyContext()
{
	for (int i = 0; i < contextCount; i++)
		if (_contextPool[i].isEmpty())
			return true;

	return false;
}