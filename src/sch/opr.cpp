# include <opr.hpp>

# include <ctxd.hpp>
# include <schd.hpp>

# include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
# include <cuda_runtime_api.h>

# include <chrono>
# include <iostream>
# include <unistd.h>
# include <future>
# include <cmath>

using namespace std;
using namespace chrono;
using namespace FGPRS;

double Operation::exceptionThreshold = 0.15;

string Operation::getName() { return _name; }
string Operation::getFullName() { return _fullName; }

void Operation::setName(string name)
{
	_name = name;
	_fullName = name;
}

void Operation::setParentName(string parentName)
{
	if (_lastParentName != parentName)
	{
		_fullName = parentName + "->" + _fullName;
		_lastParentName = parentName;
	}
}

Tensor Operation::analyze(int warmup, int repeat, Tensor input)
{
	NoGradGuard no_grad;
	Tensor output;
	bool first = true;
	steady_clock::time_point t1, t2, tStart, tEnd, tNow;
	int countIsolated, countOccupied;

	predictability = 0;
	isolatedScalability = 0;
	occupiedScalability = 0;

	_parent->analyzeLogger->info("{}:", _fullName);

	MyContext::selectDefault();
	contextData.clear();

	tStart = steady_clock::now();
	tEnd = tStart + milliseconds(warmup);

	for (int i = 0; i < warmup; i++)
		output = sequential->forward(input);

	for (auto sm : Scheduler::smOptions)
	{
		auto ctx = Scheduler::selectContext(sm);
		ctx->select();

		tStart = steady_clock::now();

		for (int i = 0; i < repeat; i++)
			output = sequential->forward(input);

		tNow = steady_clock::now();

		duration<double> d1 = tNow - tStart;

		ctx->release();
		Scheduler::startDummy(ctx);
		usleep(1000);
		ctx->select();

		countOccupied = 0;
		tStart = steady_clock::now();
		tEnd = tStart + milliseconds(repeat);

		tStart = steady_clock::now();

		for (int i = 0; i < repeat; i++)
			output = sequential->forward(input);

		tNow = steady_clock::now();

		duration<double> d2 = tNow - tStart;

		Scheduler::stopDummy();
		ctx->release();

		contextData.push_back(ContextData(ctx, d1.count() / repeat * 1000000, d2.count() / repeat * 1000000));
		_parent->analyzeLogger->info("\t{}\t{:.0f}us, {:.0f}us",
			ctx->smCount, contextData.back().isolatedExecutionTime, contextData.back().occupiedExecutionTime);

		if (!first)
		{
			contextData.back().isolatedExecutionTime =
				min(contextData.back().isolatedExecutionTime, contextData[contextData.size() - 2].isolatedExecutionTime);
			contextData.back().occupiedExecutionTime =
				min(contextData.back().occupiedExecutionTime, contextData[contextData.size() - 2].occupiedExecutionTime);
		}

		predictability += 1 - (contextData.back().occupiedExecutionTime - contextData.back().isolatedExecutionTime) / contextData.back().occupiedExecutionTime;

		if (first)
		{
			first = false;
			continue;
		}

		double desired, isolatedGain, occupiedGain;

		desired = (double)contextData.back().smCount / contextData.end()[-2].smCount;
		isolatedGain = contextData.end()[-2].isolatedExecutionTime / contextData.back().isolatedExecutionTime;
		occupiedGain = contextData.end()[-2].occupiedExecutionTime / contextData.back().occupiedExecutionTime;

		isolatedScalability += max((isolatedGain - 1) / (desired - 1), 0.0);
		occupiedScalability += max((occupiedGain - 1) / (desired - 1), 0.0);
	}

	predictability /= 4;
	isolatedScalability /= 3;
	occupiedScalability /= 3;

	_parent->analyzeLogger->info("Params: {:.2f}\t{:.2f}\t{:.2f}", predictability, isolatedScalability, occupiedScalability);

	return output;
}

vector<Tensor> Operation::analyzeSIMO(int warmup, int repeat, Tensor input)
{
	NoGradGuard no_grad;
	vector<Tensor> output;
	bool first = true;
	steady_clock::time_point t1, t2, tStart, tEnd, tNow;
	int countIsolated, countOccupied;

	predictability = 0;
	isolatedScalability = 0;
	occupiedScalability = 0;

	_parent->analyzeLogger->info("{}:", _fullName);

	MyContext::selectDefault();
	contextData.clear();

	tStart = steady_clock::now();
	tEnd = tStart + milliseconds(warmup);

	for (int i = 0; i < warmup; i++)
		output = container->forwardSIMO(input);

	for (auto sm : Scheduler::smOptions)
	{
		auto ctx = Scheduler::selectContext(sm);
		ctx->select();

		tStart = steady_clock::now();

		for (int i = 0; i < repeat; i++)
			output = container->forwardSIMO(input);

		tNow = steady_clock::now();

		duration<double> d1 = tNow - tStart;

		ctx->release();
		Scheduler::startDummy(ctx);
		usleep(1000);
		ctx->select();

		countOccupied = 0;
		tStart = steady_clock::now();
		tEnd = tStart + milliseconds(repeat);

		tStart = steady_clock::now();

		for (int i = 0; i < repeat; i++)
			output = container->forwardSIMO(input);

		tNow = steady_clock::now();

		duration<double> d2 = tNow - tStart;

		Scheduler::stopDummy();
		ctx->release();

		contextData.push_back(ContextData(ctx, d1.count() / repeat * 1000000, d2.count() / repeat * 1000000));
		_parent->analyzeLogger->info("\t{}\t{:.0f}us, {:.0f}us",
			ctx->smCount, contextData.back().isolatedExecutionTime, contextData.back().occupiedExecutionTime);

		if (!first)
		{
			contextData.back().isolatedExecutionTime =
				min(contextData.back().isolatedExecutionTime, contextData[contextData.size() - 2].isolatedExecutionTime);
			contextData.back().occupiedExecutionTime =
				min(contextData.back().occupiedExecutionTime, contextData[contextData.size() - 2].occupiedExecutionTime);
		}

		predictability += 1 - (contextData.back().occupiedExecutionTime - contextData.back().isolatedExecutionTime) / contextData.back().occupiedExecutionTime;

		if (first)
		{
			first = false;
			continue;
		}

		double desired, isolatedGain, occupiedGain;

		desired = (double)contextData.back().smCount / contextData.end()[-2].smCount;
		isolatedGain = contextData.end()[-2].isolatedExecutionTime / contextData.back().isolatedExecutionTime;
		occupiedGain = contextData.end()[-2].occupiedExecutionTime / contextData.back().occupiedExecutionTime;

		isolatedScalability += max((isolatedGain - 1) / (desired - 1), 0.0);
		occupiedScalability += max((occupiedGain - 1) / (desired - 1), 0.0);
	}

	predictability /= 4;
	isolatedScalability /= 3;
	occupiedScalability /= 3;

	_parent->analyzeLogger->info("Params: {:.2f}\t{:.2f}\t{:.2f}", predictability, isolatedScalability, occupiedScalability);

	return output;
}

void thrdFunction(Operation* operation, Tensor* input)
{
	// _chosenContext = Scheduler::getMinimalContext(this);
	// // _chosenContext = Scheduler::getFastestContext(this);
	// _chosenContext->queueOperation(operation);

	// *input = operation->sequential->forward(*input);

	// _chosenContext->dequeueOperation(operation);

	*input = operation->scheduleSync(*input);
}

void Operation::start(Tensor* input)
{
	_output = new Tensor();
	*_output = *input;

	_th = thread(thrdFunction, this, _output);
}

Tensor Operation::getResult()
{
	if (!_isException)
		_th.join();

	return *_output;
}
mutex mLock;

// Tensor Operation::runSync(Tensor input, c10::cuda::CUDAStream* stream)
Tensor Operation::runSync(Tensor input)
{
	// cudaStream_t strm;
	// cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking);
	// auto stream = c10::cuda::CUDAStream(strm);
	// // auto stream = at::cuda::createCUDAStream();
	// mLock.lock();
	// mLock.unlock();
	// std::this_thread::sleep_for(std::chrono::milliseconds(1));
	_stream = at::cuda::getStreamFromPool(highPriority, _chosenContext->index);
	at::cuda::setCurrentCUDAStream(_stream);
	return sequential->forward(input);
	// return input + input;
}

// vector<Tensor> Operation::runSIMOSync(Tensor input, c10::cuda::CUDAStream* stream)
vector<Tensor> Operation::runSIMOSync(Tensor input)
{
	_stream = at::cuda::getStreamFromPool(highPriority, _chosenContext->index);
	at::cuda::setCurrentCUDAStream(_stream);
	return container->forwardSIMO(input);
}

void Operation::startSchedule(Tensor* input)
{
	auto now = steady_clock::now();

	if (false)//occupiedScalability < exceptionThreshold)
	{
		// _isException = true;
		// _chosenContext = Scheduler::selectDefaultContext();

		// _chosenContext->select();
		// _chosenContext->lock();
		// _chosenContext->queueOperation(this);

		// runSync(*input);

		// _chosenContext->unlock();
		// _chosenContext->release();
		// _chosenContext->dequeueOperation(operation);
	}

	else
	{
		_isException = false;
		start(input);
	}
}

Tensor Operation::scheduleSync(Tensor input)
{
	startTime = steady_clock::now();

	if (false)//occupiedScalability < exceptionThreshold)
	{
		_isException = true;
		_chosenContext = Scheduler::selectDefaultContext();
		_chosenContext->queueOperation(this);
		_chosenContext->lock();
	}

	else
	{
		_isException = false;
		_chosenContext = Scheduler::getMinimalContext(this);
		// _chosenContext = Scheduler::getFastestContext(this);
	}

	_chosenContext->select();

	_parent->scheduleLogger->info("Start  {}: {} SMs -> {}",
		_fullName.c_str(), _chosenContext->smCount, _chosenContext->queue.size());

	// _parent->scheduleLogger->info("Start  {}: {} SMs -> {}\t\t\tMemory Before: {}GB",
	// 	_fullName.c_str(), _chosenContext->smCount, _chosenContext->queue.size(), Scheduler::getFreeMemoryGB());

	input = runSync(input);

	// _parent->scheduleLogger->info(
	// 	"End    {}: {} ({})-> {} + {} = {} ({})",
	// 	_fullName.c_str(),
	// 	(int)contextData[_chosenContext->index].occupiedExecutionTime,
	// 	duration_cast<microseconds>(finishTime - startTime).count(),
	// 	duration_cast<microseconds>(steady_clock::now() - startTime).count(),
	// 	duration_cast<microseconds>(absoluteDeadline - steady_clock::now()).count(),
	// 	duration_cast<microseconds>(absoluteDeadline - startTime).count(),
	// 	(long)relativeDeadline[2]);

	_parent->scheduleLogger->info(
		"End    {}: {} -> {} + {} = {} ({})\t\t\tMemory  After: {}GB",
		_fullName.c_str(),
		(int)contextData[_chosenContext->index].occupiedExecutionTime,
		duration_cast<microseconds>(steady_clock::now() - startTime).count(),
		duration_cast<microseconds>(absoluteDeadline - steady_clock::now()).count(),
		duration_cast<microseconds>(absoluteDeadline - startTime).count(),
		(long)relativeDeadline[2], Scheduler::getFreeMemoryGB());

	// _chosenContext->release();
	mLock.lock();
	_chosenContext->dequeueOperation(this);
	_chosenContext->unlock();
	mLock.unlock();

	return input;
}

vector<Tensor> Operation::scheduleSIMOSync(Tensor input)
{
	startTime = steady_clock::now();

	if (false)//occupiedScalability < exceptionThreshold)
	{
		_isException = true;
		_chosenContext = Scheduler::selectDefaultContext();
		_chosenContext->queueOperation(this);
		_chosenContext->lock();
	}

	else
	{
		_isException = false;
		_chosenContext = Scheduler::getMinimalContext(this);
		// _chosenContext = Scheduler::getFastestContext(this);
	}

	_chosenContext->select();

	_parent->scheduleLogger->info("Start  {}: {} SMs -> {}\t\t\tMemory Before: {}GB",
		_fullName.c_str(), _chosenContext->smCount, _chosenContext->queue.size(), Scheduler::getFreeMemoryGB());

	auto output = runSIMOSync(input);

	_parent->scheduleLogger->info(
		"End    {}: {} -> {} + {} = {} ({})\t\t\tMemory  After: {}GB",
		_fullName.c_str(),
		(int)contextData[_chosenContext->index].occupiedExecutionTime,
		duration_cast<microseconds>(steady_clock::now() - startTime).count(),
		duration_cast<microseconds>(absoluteDeadline - steady_clock::now()).count(),
		duration_cast<microseconds>(absoluteDeadline - startTime).count(),
		(long)relativeDeadline[2], Scheduler::getFreeMemoryGB());

	_chosenContext->release();
	_chosenContext->dequeueOperation(this);
	_chosenContext->unlock();

	return output;
}

double Operation::getRegulatedExecutionTime(int contextIndex)
{
	return contextData[contextIndex].occupiedExecutionTime;// *max(1 - occupiedScalability, 0.25);
}

void Operation::setAbsoluteDeadline(int level, steady_clock::time_point start)
{
	absoluteDeadline = start + microseconds((int)stackedDeadline[level - 1]);
	// cout << level << endl;
	// cout << getFullName() << "->" << stackedDeadline[level - 1] << endl;
}