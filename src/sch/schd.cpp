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

using namespace FGPRS;

using namespace std;

int Scheduler::maxSmCount;
bool Scheduler::_stopDummy;
vector<int> Scheduler::smOptions;
MyContext* Scheduler::_contextPool;
MyContext Scheduler::_defaultContext;

Sequential Scheduler::_dummyModule[3];
Tensor Scheduler::_dummyInput[3];

bool Scheduler::initialize(int options[], int size)
{
	bool result = true;
	cudaDeviceProp prop;

	cudaGetDeviceProperties(&prop, 0);
	maxSmCount = prop.multiProcessorCount;

	_defaultContext = MyContext(maxSmCount, -1, true);
	_contextPool = new MyContext[size];

	for (int i = 0; i < (size - 1); i++)
	{
		_dummyInput[i] = torch::randn({ 1, 3, 900, 1600 }, kCUDA);
		_dummyModule[i] = Sequential(
			Conv2d(Conv2dOptions(3, 16, 7).stride(2).padding(3)),
			BatchNorm2d(16),
			ReLU(),
			MaxPool2d(MaxPool2dOptions(2))
		);

		_dummyModule[i]->eval();
		_dummyModule[i]->to(kCUDA);
	}

	for (int i = 0; i < size; i++)
	{
		smOptions.push_back(max(options[i], 1));
		_contextPool[i] = MyContext(options[i], i);
		result &= _contextPool[i].initialize();
	}

	selectDefaultContext();

	return result;
}

MyContext* Scheduler::selectContext(int smCount)
{
	for (int i = 0; i < smOptions.size(); i++)
		if (_contextPool[i].smCount >= smCount && !_contextPool[i].busy)
			return &_contextPool[i];

	for (int i = 0; i < smOptions.size(); i++)
		if (!_contextPool[i].busy)
			return &_contextPool[i];

	for (int i = 0; i < smOptions.size(); i++)
		cout << "\t" << _contextPool[i].smCount << ", " << _contextPool[i].busy << endl;

	return &_defaultContext;
}

MyContext* Scheduler::selectContextByIndex(int index)
{
	return &_contextPool[index];
}

MyContext* Scheduler::selectDefaultContext()
{
	return &_defaultContext;
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

mutex mtx;

void Scheduler::dummyFunction(MyContext* ctx, Sequential* mod, Tensor* in)
{
	int counter = 0;
	ctx->select();

	while (!_stopDummy)
	{
		auto dummy = (*mod)->forward(*in);
		counter++;
	}

	ctx->release();
}

future<void> Scheduler::_th[3];

void Scheduler::startDummy(MyContext* ctx)
{
	int index = 0;
	_stopDummy = false;

	for (int i = 0; i < smOptions.size(); i++)
	{
		if (_contextPool[i].index == ctx->index)
			continue;

		Scheduler::selectDefaultContext();
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

	for (int i = 0; i < 3; i++)
		_th[i].get();
}

mutex globalMutex;

MyContext* Scheduler::getBestContext(Operation* operation)
{
	MyContext* ctx1 = NULL, * ctx2;
	globalMutex.lock();

	steady_clock::time_point earliest1 = steady_clock::now() + seconds(1), earliest2 = steady_clock::now() + seconds(1);
	steady_clock::time_point temp;

	for (int i = 0; i < smOptions.size(); i++)
	{
		if (!_contextPool[i].isEmpty())
			continue;

		temp = _contextPool[i].getFinishTime() + microseconds((int)operation->contextData[i].occupiedExecutionTime);

		if (temp < operation->absoluteDeadline)
		{
			_contextPool[i].queueOperation(operation);
			_contextPool[i].lock();
			globalMutex.unlock();
			return &_contextPool[i];
		}
	}

	for (int i = 0; i < smOptions.size(); i++)
	{
		temp = _contextPool[i].getFinishTime() + microseconds((int)operation->contextData[i].occupiedExecutionTime);

		if (temp < operation->absoluteDeadline)
		{
			_contextPool[i].queueOperation(operation);
			_contextPool[i].lock();
			globalMutex.unlock();
			return &_contextPool[i];
		}

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

	cout << "Zorake!\n";

	if (ctx1 != NULL)
	{
		ctx1->queueOperation(operation);
		ctx1->lock();

		globalMutex.unlock();
		return ctx1;
	}


	else
	{
		ctx2->queueOperation(operation);
		ctx2->lock();

		globalMutex.unlock();
		return ctx2;
	}
}