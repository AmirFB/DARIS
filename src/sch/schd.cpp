# include <schd.h>

# include <iostream>
# include <thread>
# include <future>
# include <ranges>
# include <vector>

# include <cuda.h>
# include <cudaTypedefs.h>
# include <cuda_runtime.h>

using namespace FGPRS;

using namespace std;

int Scheduler::maxSmCount;
bool Scheduler::_stopDummy;
vector<int> Scheduler::smOptions;
vector<MyContext> Scheduler::_contextPool;
MyContext Scheduler::_defaultContext;
Sequential Scheduler::_dummyModule(Conv2d(Conv2dOptions(3, 7, 7).stride(2).padding(1)));
Tensor Scheduler::_dummyInput = torch::randn({3, 2048, 2048}, kCUDA);
Sequential Scheduler::_dummyModule2(Linear(256, 1000));
Tensor Scheduler::_dummyInput2 = torch::randn(256, kCUDA);

bool Scheduler::initialize(int options[], int size)
{
	bool result = true;
	cudaDeviceProp prop;

	cudaGetDeviceProperties(&prop, 0);
	maxSmCount = prop.multiProcessorCount;

	for (int i = 0; i < size; i++)
	{
		smOptions.push_back(max(options[i], 1));
		_contextPool.push_back(MyContext(options[i]));
		result &= _contextPool.back().initialize();
	}

	_defaultContext = MyContext(maxSmCount, true);
	return result;
}

MyContext Scheduler::selectContext(int smCount)
{
	for (auto context : _contextPool)
		if (context.smCount >= smCount && !context.busy)
			return context;

	for (auto context : _contextPool)// | views::reverse)
		if (!context.busy)
			return context;
	
	// cout << "Going NULL: " << smCount << ", " << poolSize << endl;

	for (auto context : _contextPool)
		cout << "\t" << context.smCount << ", " << context.busy << endl;

	return _defaultContext;
}

MyContext Scheduler::selectContextByIndex(int index)
{
	return _contextPool[index];
}

MyContext Scheduler::selectContextByQueueLoad(double* executionTime)
{
	double min = _contextPool[0].queueDuration + executionTime[_contextPool[0].smCount] * 1000000;
	MyContext output = _contextPool[0];

	for (auto context : _contextPool)
	{
		if ((context.queueDuration + executionTime[context.smCount] * 1000000) < min)
		{
			output = context;
			min = context.queueDuration + executionTime[context.smCount] * 1000000;
		}
	}

	return output;
}

bool Scheduler::selectDefaultContext()
{
	return MyContext::selectDefault();
}

bool Scheduler::releaseContext(MyContext context)
{
	return context.release();
}

bool Scheduler::destroyAll()
{
	bool result = true;

	// for (int i = 0; i < (poolSize - 1); i++)
	// 	result &= _contextPool[i].destroy();
	
	return true;
}

vector<MyContext> Scheduler::getAllContexts()
{
	return _contextPool;
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

void Scheduler::dummyFunction(MyContext ctx)
{
	ctx.select();

	while (!_stopDummy)
	{
		auto dummy = _dummyModule->forward(_dummyInput);
		dummy = _dummyModule2->forward(_dummyInput2);
	}

	ctx.release();
}

bool first = true;
future<void> th;

int Scheduler::startDummy(int sms)
{
	if (first)
	{
		first = false;
		_dummyModule->eval();
		_dummyModule->to(kCUDA);
		_dummyModule2->eval();
		_dummyModule2->to(kCUDA);
	}

	_stopDummy = false;
	auto ctx = selectContext(sms);

	Scheduler::selectDefaultContext();
	auto dummy = _dummyModule->forward(_dummyInput);
	dummy = _dummyModule2->forward(_dummyInput2);

	ctx.select();
	dummy = _dummyModule->forward(_dummyInput);
	dummy = _dummyModule2->forward(_dummyInput2);
	ctx.release();

	th = async(dummyFunction, ctx);
	return ctx.smCount;
}

void Scheduler::stopDummy()
{
	_stopDummy = true;
	th.get();
}