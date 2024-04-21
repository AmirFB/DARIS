# include <main.hpp>

# include <opr.hpp>

# include <schd.hpp>

# include <torch/torch.h>
# include <c10/cuda/CUDAStream.h>
# include <cuda_runtime_api.h>
# include <nvToolsExt.h>
# include <c10/cuda/CUDACachingAllocator.h>

# include <chrono>
# include <iostream>
# include <unistd.h>
# include <future>
# include <cmath>

using namespace std;
using namespace chrono;
using namespace FGPRS;

Operation::Operation(string name, shared_ptr<MyContainer> container, shared_ptr<SequentialImpl> module, bool isLast)
	: name(name), container(container), fullName(container->moduleName + "->" + name),
	sequential(module), isLast(isLast), highPriority(container->highPriority), _isNL(false), _parallelCount(0)
{
}

Operation::Operation(string name, shared_ptr<MyContainer> container, shared_ptr<MyModule> module, bool isLast, int parallelCount)
	: name(name), container(container), fullName(container->moduleName + "->" + name),
	module(module), isLast(isLast), highPriority(container->highPriority), _isNL(true), _parallelCount(parallelCount)
{
}

void Operation::setAbsoluteDeadline(steady_clock::time_point start)
{
	absoluteDeadline = start + microseconds((int)stackedDeadline);
}

Tensor Operation::forward(Tensor input)
{
	if (!_isNL)
		return sequential->forward(input);

	return module->forward(input);
}

Tensor Operation::forward(Tensor input, MyContext* ctx, MyStream* str)
{
	if (!_isNL)
		return sequential->forward(input);

	if (!container->highPriority)
		return module->forward(input);

	ctx->remainingSecondaryStreams -= _parallelCount - 1;
	auto output = module->forwardNL(input, ctx, str);
	ctx->remainingSecondaryStreams += _parallelCount - 1;

	return output;
}

Tensor Operation::analyze(int warmup, int repeat, Tensor input, int* timer)
{
	Tensor output;
	auto str = container->currentContext->getStream();
	str->select();

	for (int i = 0; i < warmup; i++)
	{
		output = forward(input);
		str->synchronize();
	}

	auto start = steady_clock::now();

	for (int i = 0; i < repeat; i++)
	{
		output = forward(input);
		str->synchronize();
	}

	auto end = steady_clock::now();

	auto duration = duration_cast<microseconds>(end - start).count();
	*timer = duration / repeat;

	str->release();
	cudaDeviceSynchronize();

	return output;
}

Tensor Operation::releaseSync(Tensor input)
{
	auto ctx = container->currentContext;
	// cout << "Acquired " << fullName << endl;
	ctx->releaseOperation(shared_from_this());
	// cout << "Streaming " << fullName << endl;
	auto str = ctx->getStream();

	str->select();

	// auto id = nvtxRangeStart(fullName.c_str());
	auto start = steady_clock::now();
	// cout << "Start " << fullName << endl;
	input = forward(input, ctx, str);
	// cout << "Forwarded " << fullName << endl;
	str->release();
	// cout << "Released " << fullName << endl;
	ctx->finishOperation(shared_from_this());

	auto end = steady_clock::now();
	delayed = end > absoluteDeadline;
	// nvtxRangeEnd(id);

	auto timer = duration_cast<microseconds>(end - start).count();

	finished = true;

	container->currentRecord->setOperationExecutionTime(this, delayed, timer);

	return input;
}

future<Tensor> Operation::releaseAsync(Tensor input)
{
	return async(launch::async, &Operation::releaseSync, this, input);
}