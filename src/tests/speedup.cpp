# include <stdio.h>
# include <stdlib.h>
# include <sys/stat.h>

# include <iostream>
# include <chrono>

# include <torch/torch.h>
# include <torch/script.h>
# include <c10/cuda/CUDAStream.h>
# include <ATen/cuda/CUDAContext.h>
# include <c10/cuda/CUDACachingAllocator.h>

# include <cuda.h>
# include <cudaTypedefs.h>
# include <cuda_runtime.h>

# include <tests.hpp>
// # include <ctx.hpp>
# include <schd.hpp>

using namespace std;
using namespace std::chrono;
using namespace torch;
using namespace torch::nn;
using namespace FGPRS;

# define MODULE_COUNT	8

void testSpeedup(char** argv)
{
	string moduleName[] = {
		"CV", "FC", "BN", "RL", "MP", "AP", "DP", "DA" };
	NoGradGuard no_grad;
	Sequential mod[MODULE_COUNT];
	Tensor in[MODULE_COUNT];

	int smCount, timer;

	smCount = atoi(argv[0]);
	timer = (int)(atof(argv[1]) * 1000);

	printf("Running \"Speedup\" simulation. (SM count: %d)\n", smCount);

	mod[0] = Sequential(Conv2d(Conv2dOptions(512, 1024, 3).stride(2).padding(1)));
	in[0] = torch::randn({ 512, 48, 48 }, kCUDA);

	// mod[1] = Sequential(Conv2d(Conv2dOptions(1024, 2048, 3).stride(2).padding(1)));
	// in[1] = torch::randn({1024, 32, 32}, kCUDA);

	// mod[2] = Sequential(Conv2d(Conv2dOptions(3, 64, 7).stride(2).padding(2)));
	// in[2] = torch::randn({3, 1024, 1024}, kCUDA);

	// mod[0] = Sequential(Linear(4096, 1000));
	// in[0] = torch::randn(4096, kCUDA);

	// mod[1] = Sequential(Linear(4096, 1000));
	// in[1] = torch::randn(4096, kCUDA);

	mod[1] = Sequential(Linear(4096 * 4, 10000));
	in[1] = torch::randn(4096 * 4, kCUDA);

	// mod[3] = Sequential(Linear(4096, 1000));
	// in[3] = torch::randn(4096, kCUDA);

	mod[2] = Sequential(BatchNorm2d(1024));
	in[2] = torch::randn({ 1, 1024, 32, 24 }, kCUDA);

	mod[3] = Sequential(ReLU());
	in[3] = torch::randn({ 1024, 32, 24 }, kCUDA);

	mod[4] = Sequential(MaxPool2d(MaxPool2dOptions(3).stride(2).padding(1)));
	in[4] = torch::randn({ 64, 224, 224 }, kCUDA);

	mod[5] = Sequential(AvgPool2d(AvgPool2dOptions(3).stride(2).padding(1)));
	in[5] = torch::randn({ 64, 224, 224 }, kCUDA);

	mod[6] = Sequential(AdaptiveMaxPool2d(AdaptiveMaxPool2dOptions(1)));
	in[6] = torch::randn({ 2048, 3, 3 }, kCUDA);

	mod[7] = Sequential(AdaptiveAvgPool2d(AdaptiveAvgPool2dOptions(1)));
	in[7] = torch::randn({ 2048, 3, 3 }, kCUDA);

	for (int i = 0; i < MODULE_COUNT; i++)
	{
		mod[i]->eval();
		mod[i]->to(kCUDA);
	}

	int options[] = { smCount };

	if (!Scheduler::initialize(options, 1))
	{
		cout << "CUDA initialization failed.\n";
		return;
	}

	cout << "-------------------------------------------------------------------------------\n";
	cout << "Warming up\n";

	auto ctx = Scheduler::selectContext(smCount);

	Tensor dummy;
	steady_clock::time_point tstart, now, tend;
	tstart = steady_clock::now();
	tend = tstart + milliseconds(timer);

	auto ctxD = Scheduler::selectContext(68);
	ctxD->select();

	while (true)
	{
		for (int j = 0; j < MODULE_COUNT; j++)
			dummy = mod[j]->forward(in[j]);

		cuCtxSynchronize();
		now = steady_clock::now();

		if (tend <= steady_clock::now())
			break;
	}

	ctxD->release();
	ctx->select();

	while (true)
	{
		for (int j = 0; j < MODULE_COUNT; j++)
			dummy = mod[j]->forward(in[j]);

		cuCtxSynchronize();
		now = steady_clock::now();

		if (tend <= steady_clock::now())
			break;
	}

	ctx->release();

	steady_clock::time_point t1, t2;
	duration<double> d;
	vector<double> results(MODULE_COUNT);

	ctx->select();

	for (int j = 0; j < MODULE_COUNT; j++)
	{
		cout << "Running operation \"" << moduleName[j] << "\": ";
		int count = 0;

		tstart = steady_clock::now();
		tend = tstart + milliseconds(timer);

		while (true)
		{
			dummy = mod[j]->forward(in[j]);
			cuCtxSynchronize();
			count++;
			now = steady_clock::now();

			if (tend <= steady_clock::now())
				break;
		}

		d = now - tstart;
		results[j] = d.count() / count * 1000000;
		printf("%6.3lfus\n", results[j]);

		// CUexecAffinityParam_v1 affinity;
		// cuCtxGetExecAffinity(&affinity, CU_EXEC_AFFINITY_TYPE_SM_COUNT);
		// cout << "Aff: " << affinity.param.smCount.val << endl;
	}

	ctx->release();
	cout << "Saving results\n";
	writeToFile("speedup", smCount, results);
	cout << "-------------------------------------------------------------------------------\n\n";
}