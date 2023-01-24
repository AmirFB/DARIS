# include <iostream>
# include <fstream>
# include <iomanip>
# include <thread>
# include <pthread.h>
# include <chrono>
# include <string>
# include <cstdlib>
# include <future> 
# include <sys/stat.h>

# include <torch/torch.h>
# include <torch/script.h>
# include <c10/cuda/CUDAStream.h>
# include <ATen/cuda/CUDAContext.h>
# include <c10/cuda/CUDACachingAllocator.h>

# include <cuda.h>
# include <cudaTypedefs.h>
# include <cuda_runtime.h>

# include "ctx.h"
# include "schd.h"
# include "container.h"
# include "mynet.h"

# include "cif10.h"
# include "operation.h"
# include "resnet.h"

# include "tests.h"

# include <stdio.h>
# include <stdlib.h>
# include <unistd.h>

using namespace std;
using namespace std::chrono;
using namespace torch;
using namespace torch::nn;
using namespace FGPRS;

void interference_single(MyContext **ctx, Sequential model, Tensor input, int repeat, const char* name)
{
	auto t1 = steady_clock::now();
	auto t2 = steady_clock::now();
	duration<double> d;

	cout << name << endl;

	for (int i = 0; i < 8; i++)
	{
		ctx[i]->select(0);

		Tensor output;
		t1 = steady_clock::now();

		for (int j = 0; j < repeat; j++)
		{
			output = model->forward(input);
			cuCtxSynchronize();
		}

		t2 = steady_clock::now();
		ctx[i]->release(0);
		d = t2 - t1;

		cout << "\t" << ctx[i]->smCount << " SMs: " << d.count() / repeat * 1000000 << endl;
	}

	cout << endl;
}

void interference_double_thrd(
	MyContext *ctx, Sequential model, Tensor input, steady_clock::time_point tend, const char* name)
{
	int count = 0;
	Tensor out;
	steady_clock::time_point tstart, now;
	duration<double> d;

	tstart = steady_clock::now();
	ctx->select(0);

	while (true)
	{
		out = model->forward(input);
		cuCtxSynchronize();
		now = steady_clock::now();

		if (tend <= now)
			break;

		count++;
	}

	d = now - tstart;
	ctx->release(0);

	stringstream s;
	s << "\t" << name << ": " << d.count() / count * 1000000 << " us" << endl;
	cout << s.str();
}

void interference_double(int sec,
	MyContext* ctx1, Sequential model1, Tensor input1, const char* name1,
	MyContext* ctx2, Sequential model2, Tensor input2, const char* name2)
{
	int count = 0;
	Tensor out;
	steady_clock::time_point tend = steady_clock::now() + seconds(sec);

	cout << name1 << "(" << ctx1->smCount << " SMs) vs "
			 << name2 << "(" << ctx2->smCount << " SMs)" << endl;

	thread th1(interference_double_thrd, ctx1, model1, input1, tend, name1);
	thread th2(interference_double_thrd, ctx2, model2, input2, tend, name2);

	th1.join();
	th2.join();

	cout << endl;
}

void test_interference()
{
	auto warmup = 50, repeat = 500;

	auto t1 = steady_clock::now();
	auto t2 = steady_clock::now();
	auto tend = steady_clock::now();
	duration<double> d;

	auto conv1 = Sequential(Conv2d(Conv2dOptions(3, 6, 3).stride(1).padding(1)));
	conv1->eval();
	conv1->to(kCUDA);
	auto inc1 = torch::randn({3, 1024, 1024}, kCUDA);

	auto conv2 = Sequential(Conv2d(Conv2dOptions(8, 16, 5).stride(1).padding(1)));
	conv2->eval();
	conv2->to(kCUDA);
	auto inc2 = torch::randn({8, 256, 256}, kCUDA);

	int inputSize1 = 8192 * 2;
	auto lin1 = Sequential(Linear(inputSize1, inputSize1 /2));
	lin1->eval();
	lin1->to(kCUDA);
	auto inl1 = torch::rand(inputSize1, kCUDA);

	int inputSize2 = 2048;
	auto lin2 = Sequential(Linear(inputSize2, inputSize2));
	lin2->eval();
	lin2->to(kCUDA);
	auto inl2 = torch::rand(inputSize2, kCUDA);

	int options[] = {2, 18, 34, 34, 50, 64, 68};

	if (!Scheduler::initialize(options, 7))
	{
		cout << "Initialization failed!" << endl;
		// return;
	}

	MyContext *ctx[8];

	for (int i = 0; i < 7; i ++)
	{
		ctx[i] = Scheduler::selectContext(options[i]);
		ctx[i]->select(0);
	}

	ctx[7] = Scheduler::selectContext(68);

	for (int i = 0; i < 7; i ++)
		ctx[i]->release(0);

	for (int i = 0; i < 8; i++)
	{
		Tensor output;

		ctx[i]->select(0);

		for (int j = 0; j < warmup; j++)
		{
			output = conv1->forward(inc1);
			output = conv2->forward(inc2);
			output = lin1->forward(inl1);
			output = lin2->forward(inl2);

			cuCtxSynchronize();
		}

		ctx[i]->release(0);
	}

	interference_single(ctx, conv1, inc1, repeat, "CNV1");
	interference_single(ctx, conv2, inc2, repeat, "CNV2");
	interference_single(ctx, lin1, inl1, repeat, "LIN1");
	interference_single(ctx, lin2, inl2, repeat, "LIN2");

	// interference_double(ctx[6], conv1, inc1, "CNV1", ctx[7], conv2, inc2, "CNV2");
	// interference_double(ctx[6], conv1, inc1, "CNV1", ctx[6], conv2, inc2, "CNV2");
	// interference_double(ctx[2], conv1, inc1, "CNV1", ctx[3], conv2, inc2, "CNV2");
	// interference_double(ctx[1], conv1, inc1, "CNV1", ctx[0], conv2, inc2, "CNV2");
	// interference_double(ctx[5], conv1, inc1, "CNV1", ctx[0], conv2, inc2, "CNV2");
	// interference_double(ctx[4], conv1, inc1, "CNV1", ctx[1], conv2, inc2, "CNV2");
	// interference_double(ctx[4], conv1, inc1, "CNV1", ctx[2], conv2, inc2, "CNV2");
	// interference_double(ctx[5], conv1, inc1, "CNV1", ctx[1], conv2, inc2, "CNV2");

	// interference_double(ctx[0], lin1, inl1, "LIN1", ctx[1], lin2, inl2, "LIN2");
	// interference_double(ctx[2], lin1, inl1, "LIN1", ctx[3], lin2, inl2, "LIN2");
	// interference_double(ctx[4], lin1, inl1, "LIN1", ctx[5], lin2, inl2, "LIN2");
	// interference_double(ctx[1], lin1, inl1, "LIN1", ctx[1], lin2, inl2, "LIN2");
	// interference_double(ctx[0], lin1, inl1, "LIN1", ctx[0], lin2, inl2, "LIN2");
	// interference_double(ctx[5], lin1, inl1, "LIN1", ctx[0], lin2, inl2, "LIN2");
	// interference_double(ctx[4], lin1, inl1, "LIN1", ctx[1], lin2, inl2, "LIN2");
	// interference_double(ctx[4], lin1, inl1, "LIN1", ctx[2], lin2, inl2, "LIN2");
	// interference_double(ctx[5], lin1, inl1, "LIN1", ctx[1], lin2, inl2, "LIN2");

	// interference_double(ctx[6], conv1, inc1, "CNV1", ctx[6], lin1, inl1, "LIN1");
	// interference_double(ctx[6], conv1, inc1, "CNV1", ctx[7], lin1, inl1, "LIN1");
	// interference_double(ctx[2], conv1, inc1, "CNV1", ctx[3], lin1, inl1, "LIN1");
	// interference_double(ctx[4], conv1, inc1, "CNV1", ctx[1], lin1, inl1, "LIN1");
	// interference_double(ctx[5], conv1, inc1, "CNV1", ctx[0], lin1, inl1, "LIN1");
	// interference_double(ctx[6], conv1, inc1, "CNV1", ctx[0], lin1, inl1, "LIN1");
	// interference_double(ctx[6], conv1, inc1, "CNV1", ctx[1], lin1, inl1, "LIN1");
	// interference_double(ctx[6], conv1, inc1, "CNV1", ctx[2], lin1, inl1, "LIN1");
	// interference_double(ctx[1], conv1, inc1, "CNV1", ctx[0], lin1, inl1, "LIN1");
	// interference_double(ctx[4], conv1, inc1, "CNV1", ctx[0], lin1, inl1, "LIN1");
	// interference_double(ctx[5], conv1, inc1, "CNV1", ctx[0], lin1, inl1, "LIN1");
}

int main(int argc, char** argv)
{
	char *op = argv[1];
	mkdir("results", 0777 );

	if (!strcmp(op, "clear"))
	{
		cout << "Removing previous results of \"" << argv[2] << "\" simulation\n";
		remove((string("results/") + string(argv[2]) + ".csv").c_str());
	}

	else if (!strcmp(op, "speedup"))
		testSpeedup(&argv[2]);

	// else if (!strcmp(op, "concurrency"))
	// 	testConcurrency(&argv[2]);

	else if (!strcmp(op, "tailing"))
		testTailing(&argv[2]);

	else if (!strcmp(op, "interference"))
		testInterference(&argv[2]);
}