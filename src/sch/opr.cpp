# include <main.hpp>

# include <opr.hpp>

# include <ctxd.hpp>
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
	sequential(module), isLast(isLast), highPriority(container->highPriority)
{
}

void Operation::setAbsoluteDeadline(steady_clock::time_point start)
{
	absoluteDeadline = start + microseconds((int)stackedDeadline);
}

Tensor Operation::analyze(int warmup, int repeat, Tensor input, int* timer)
{
	Tensor output;
	auto str = container->currentContext->getStream();
	str->select();

	for (int i = 0; i < warmup; i++)
	{
		output = sequential->forward(input);
		str->synchronize();
	}

	auto start = steady_clock::now();

	for (int i = 0; i < repeat; i++)
	{
		output = sequential->forward(input);
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

	ctx->releaseOperation(shared_from_this());
	auto str = ctx->getStream();

	str->select();

	auto id = nvtxRangeStart(fullName.c_str());
	auto start = steady_clock::now();
	input = sequential->forward(input);
	str->release();
	ctx->finishOperation(shared_from_this());

	auto end = steady_clock::now();
	delayed = steady_clock::now() > absoluteDeadline;
	nvtxRangeEnd(id);

	auto timer = duration_cast<microseconds>(end - start).count();

	finished = true;

	container->currentRecord->setOperationExecutionTime(this, delayed, timer);

	return input;
}

future<Tensor> Operation::releaseAsync(Tensor input)
{
	return async(launch::async, &Operation::releaseSync, this, input);
}