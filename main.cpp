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
# include <errno.h>

# define _UNIX03_THREADS 1
# include <limits.h>                                                            
# include <errno.h> 

using namespace std;
using namespace std::chrono;
using namespace torch;
using namespace torch::nn;
using namespace FGPRS;

#define handle_error_en(en, msg) \
	do { errno = en; perror(msg); } while (0)
	// do { errno = en; perror(msg); exit(EXIT_FAILURE); } while (0)

void test_container()
{
	int options[] = {2, 10};
	bool result = Scheduler::initialize(options, 2);

	cout << "Initialization Result: " << result << endl;

	Scheduler::selectDefaultContext();

	// const std::vector<int64_t> inputSize{1, 3, 512, 512};
	// auto dummyInput = torch::rand(inputSize, torch::kCUDA);
	int inputSize = 256;
	auto dummyInput = torch::rand(inputSize, kCUDA);
	MyNet model{inputSize};
	auto net = &model;
	net->eval();
	net->to(torch::kCUDA);
	net->analyze(&dummyInput, 10, 100);
	cout << "Average Inference Time: " << net->executionTime << endl;

	net->analyze(&dummyInput, 10, 100);
	cout << "Average Inference Time: " << net->executionTime << endl;

	net->analyze(&dummyInput, 10, 100);
	cout << "Average Inference Time: " << net->executionTime << endl;
}

void test_speedup2()
{
	torch::NoGradGuard no_grad;
	int warmup = 100, repeat = 1000;

	// int inputSize = 8192;
	// auto input = torch::rand(inputSize, kCUDA);
	// MyNet model{inputSize};

	// auto model = resnet34(10);
	// model->eval();
	// model->to(torch::kCUDA);
	// const std::vector<int64_t> inputSize{1, 3, 300, 300};
	// auto input = torch::rand(inputSize, torch::kCUDA);

	// auto model = Conv2d(nn::Conv2dOptions(3, 3, 3).stride(1).padding(1));
	// model->eval();
	// model->to(torch::kCUDA);
	// auto input = torch::randn({3, 64, 64}, torch::kCUDA);

	// int inputSize = 8192 * 2;
	// auto model = Linear(inputSize, inputSize / 2);
	// model->eval();
	// model->to(torch::kCUDA);
	// auto input = torch::rand(inputSize, kCUDA);
  
	// auto model = BatchNorm2d(128);
	// model->eval();
	// model->to(torch::kCUDA);
	// auto input = torch::rand({ 1, 128, 48, 48 }, kCUDA);

	auto model = MaxPool2d(MaxPool2dOptions(3).stride(2).padding(1));
	model->eval();
	model->to(torch::kCUDA);
	auto input = torch::rand({ 3, 512, 512 }, kCUDA);

	// auto net = &model;
	auto net = model;
	net->eval();
	net->to(torch::kCUDA);
	double results[34];

	int options[34];

	for (int i = 0; i < 34; i++)
		options[i] = (i + 1) * 2;

	auto result = Scheduler::initialize(options, 34);

	if (!result)
		cout << "Initizalization failed." << endl;

	for (int i = 0; i < 34; i ++)
	{
		// int options[] = {i};
		// auto result = Scheduler::initialize(options, 1);
		auto ctx = Scheduler::selectContext(options[i]);
		ctx->select(0);

		for (int j = 0; j < warmup; j++)
		{
			auto output = net->forward(input);
			cuCtxSynchronize();
		}

		auto t1 = steady_clock::now();

		for (int j = 0; j < repeat; j++)
		{
			auto output = net->forward(input);
			cuCtxSynchronize();
		}

		auto t2 = steady_clock::now();
		// Scheduler::destroyAll();
		ctx->release(0);
		ctx->destroy();

		duration<double> d = t2 - t1;

		// cout << "Made it!\n";
		results[i] = d.count() / repeat * 1000000;
		cout << options[i] << ": " << results[i] << endl;
	}

	for (int i = 0; i < 34; i++)
		cout << results[i] << ", ";

	cout << endl;
}

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

void test_parallel()
{
	const int nNet = 10;
	const int minSize = 128, maxSize = 4096;
	// MyNet* pNets = new MyNet[nNet];
	int smOptions[] = {4, 8};
	auto abc{10};
	MyNet* pNets = new MyNet[nNet];
	thread ths[nNet];

	// ths = (thread*)malloc(sizeof(thread) * nNet);

	// pNets = (MyNet*)malloc(sizeof(MyNet) * nNet);

	Scheduler::initialize(smOptions, 2);

	for (int i = 0; i < nNet; i++)
	{
		int size = minSize + (maxSize - minSize) / (nNet - 1) * i;
		pNets[i].setSize(size);
		pNets[i].eval();
		pNets[i].to(kCUDA);
		Tensor dummy = torch::rand(size, kCUDA);
		pNets[i].analyze(&dummy, 10, 20);
		pNets[i].interval = (i / 2 + 1) * 2;

		cout << "Input Size: " << pNets[i].inputSize << endl;

		for (int j = 0; j < Scheduler::poolSize; j++)
		{
			cout << pNets[i].executionTime[Scheduler::selectContextByIndex(j)->smCount] << ", ";
		}

		cout << endl;
	}

	Tensor inputs[nNet];
	this_thread::sleep_for(std::chrono::milliseconds(100));

	for (int i = 0; i < nNet; i++)
	{
		cout << pNets[i].inputSize << ", " << pNets[i].interval << endl;
		inputs[i] = torch::rand(pNets[i].inputSize, kCUDA);
		/*ths[i] =*/ pNets[i].run(&inputs[i], 1000 / pNets[i].interval);
		cout << "Done: " << i << endl;
	}

	for (int i = 0; i < nNet; i++)
		pNets[i].join();
}

double speedup_double_thrd(
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

	// for (int i = 0; i < 1000; i++)
	// {
	// 	out = model->forward(input);
	// 	cuCtxSynchronize();
	// }

	now = steady_clock::now();
	d = now - tstart;
	double time = d.count() / count * 1000000;
	ctx->release(0);

	stringstream s;
	s << "\t" << name << ": " << time << " us" << endl;
	cout << s.str();

	return time;
}

void speedup_double(int sec,
	MyContext* ctx1, Sequential model1, Tensor input1, const char* name1, double* t1,
	MyContext* ctx2, Sequential model2, Tensor input2, const char* name2, double* t2)
{
	int count = 0;
	Tensor out;
	steady_clock::time_point tend = steady_clock::now() + seconds(sec);

	cout << name1 << "(" << ctx1->smCount << " SMs) vs "
			 << name2 << "(" << ctx2->smCount << " SMs)" << endl;

	auto th1 = async(speedup_double_thrd, ctx1, model1, input1, tend, name1);
	auto th2 = async(speedup_double_thrd, ctx2, model2, input2, tend, name2);
	
	*t1 = th1.get();
	*t2 = th2.get();

	cout << endl;
}

void test_speedup_double(int argc, char** argv)
{
# define SINGLE			1
# define DOUBLE			2
# define COMPLEMENT	3
# define WHICH			COMPLEMENT

	int warmup = 2, repeat = 10, from, to, len;

	from = atoi(argv[1]);
	to = atoi(argv[2]);

	auto conv1 = Sequential(Conv2d(Conv2dOptions(3, 3, 3).stride(1).padding(1)));
	conv1->eval();
	conv1->to(kCUDA);
	auto inc1 = torch::randn({3, 840, 840}, kCUDA);

	auto conv2 = Sequential(Conv2d(Conv2dOptions(3, 3, 3).stride(1).padding(1)));
	conv2->eval();
	conv2->to(kCUDA);
	auto inc2 = torch::randn({3, 840, 840}, kCUDA);

	int inputSize1 = 16384;
	auto lin1 = Sequential(Linear(inputSize1, inputSize1 / 2));
	lin1->eval();
	lin1->to(kCUDA);
	auto inl1 = torch::rand(inputSize1, kCUDA);

	int inputSize2 = 16384;
	auto lin2 = Sequential(Linear(inputSize2, inputSize2 / 2));
	lin2->eval();
	lin2->to(kCUDA);
	auto inl2 = torch::rand(inputSize2, kCUDA);

	int inputSize3 = 16384 * 1.25;
	auto lin3 = Sequential(Linear(inputSize3, inputSize3 / 1.5));
	lin3->eval();
	lin3->to(kCUDA);
	auto inl3 = torch::rand(inputSize3, kCUDA);

	int inputSize4 = 16384 * 0.25;
	auto lin4 = Sequential(Linear(inputSize4, inputSize4 / 2));
	lin4->eval();
	lin4->to(kCUDA);
	auto inl4 = torch::rand(inputSize4, kCUDA);
	
# if WHICH == SINGLE
	len = (to - from) / 2 - 1;
	int *options = new int[len];

	double *tc1, *tc2, *tl1, *tl2;
	tc1 = new double[len];
	tc2 = new double[len];
	tl1 = new double[len];
	tl2 = new double[len];
# elif WHICH == DOUBLE
	len = to - from + 2;
	int *options = new int[len];

	double *tc1, *tc2, *tc3, *tl1, *tl2, *tl3, *tl4, *tl5, *tl6, *tl7;
	tc1 = new double[len / 2];
	tc2 = new double[len / 2];
	tc3 = new double[len / 2];
	tl1 = new double[len / 2];
	tl2 = new double[len / 2];
	tl3 = new double[len / 2];
	tl4 = new double[len / 2];
	tl5 = new double[len / 2];
	tl6 = new double[len / 2];
	tl7 = new double[len / 2];
# elif WHICH == COMPLEMENT
	len = to - from + 2;
	int *options = new int[len];

	double *tc1, *tc2, *tc3, *tc4, *tl1, *tl2, *tl3, *tl4, *tl5, *tl6, *tl7, *tl8, *tl9;
	tc1 = new double[len / 2];
	tc2 = new double[len / 2];
	tc3 = new double[len / 2];
	tl1 = new double[len / 2];
	tl2 = new double[len / 2];
	tl3 = new double[len / 2];
	tl4 = new double[len / 2];
	tl5 = new double[len / 2];
	tl6 = new double[len / 2];
	tl7 = new double[len / 2];
	tl8 = new double[len / 2];
	tl9 = new double[len / 2];
# endif

	steady_clock::time_point t1, t2;
	duration<double> d;

# if WHICH == SINGLE
	for (int i = 0; i < len; i++)
		options[i] = from + i * 2;

	cout << "Starting speedup simulation for \"Isolation\" scenario (" << from << "-" << to << ")" << endl;
# elif WHICH == DOUBLE
	for (int i = 0; i < len; i++)
		options[i] = from + (i - i % 2);
	
	cout << "Starting speedup simulation for \"Concurrent\" scenario (" << from << "-" << to << ")" << endl;
# elif WHICH == COMPLEMENT
	for (int i = 0; i < len / 2; i++)
	{
		options[i] = from + i * 2;
		options[len - i - 1] = 68 - options[i];
	}
	
	cout << "Starting speedup simulation for \"Complement\" scenario (" << from << "-" << to << ")" << endl;
# endif

	auto result = Scheduler::initialize(options, len);

# if WHICH == SINGLE
	cout << "\t  CNV\t  LIN" << endl;

	for (int i = 0; i < len; i++)
	{
		auto ctx = Scheduler::selectContext(options[i]);

		ctx->select(options[i]);

		for (int j = 0; j < warmup; j++)
		{
			auto out = conv1->forward(inc1);
			out = conv2->forward(inc2);
			out = lin1->forward(inl1);
			out = lin2->forward(inl2);
		}

		t1 = steady_clock::now();

		for (int j = 0; j < repeat; j++)
		{
			auto out = conv1->forward(inc1);
			cuCtxSynchronize();
		}

		t2 = steady_clock::now();
		d = t2 - t1;

		tc1[i] = d.count() / repeat * 1000000;

		t1 = steady_clock::now();

		for (int j = 0; j < repeat; j++)
		{
			auto out = lin1->forward(inl1);
			cuCtxSynchronize();
		}

		t2 = steady_clock::now();
		d = t2 - t1;

		tl1[i] = d.count() / repeat * 1000000;
		cout << options[i] << ":";
		printf("\t%7.1lf\t%7.1lf\n", tc1[i], tl1[i]);
	}

	cout << "Speedup simulation for isolation case finished." << endl;

	cout << "CNV1 speedup results: " << endl;

	for (int i = 0; i < len; i++)
		cout << tc1[i] << ", ";

	cout << endl;

	for (int i = 0; i < len; i++)
		cout << tl1[i] << ", ";

	cout << endl;
# elif WHICH == DOUBLE || WHICH == COMPLEMENT
# if WHICH == DOUBLE
	for (int i = 0; i < len; i += 2)
	{
		auto ctx1 = Scheduler::selectContext(options[i]);
		ctx1->select(0);

		for (int j = 0; j < warmup; j++)
		{
			auto out = conv1->forward(inc1);
			out = lin1->forward(inl1);
		}

		auto ctx2 = Scheduler::selectContext(options[i]);
		ctx2->select(0);

		for (int j = 0; j < warmup; j++)
		{
			auto out = conv2->forward(inc2);
			out = lin2->forward(inl2);
			out = lin3->forward(inl3);
		}

		ctx1->release(0);
		ctx2->release(0);

		int sec = 20 / options[i] + 2;

		speedup_double(sec, ctx1, conv1, inc1, "CNV1", tc1 + i / 2, ctx2, conv2, inc2, "CNV2", tc2 + i / 2);
		speedup_double(sec, ctx1, conv1, inc1, "CNV", tc3 + i / 2, ctx2, lin2, inl2, "LIN", tl3 + i / 2);
		speedup_double(sec, ctx1, lin1, inl1, "LIN1", tl1 + i / 2, ctx2, lin2, inl2, "LIN2", tl2 + i / 2);
		speedup_double(sec, ctx1, lin1, inl1, "LIN1", tl4 + i / 2, ctx2, lin3, inl3, "LIN3", tl6 + i / 2);
		speedup_double(sec, ctx1, lin1, inl1, "LIN1", tl5 + i / 2, ctx2, lin4, inl4, "LIN4", tl7 + i / 2);

		cout << endl;
	}

	cout << "CNV1 vs CNV2: ";

	for (int i = 0; i < len / 2; i++)
		cout << (tc1[i] + tc2[i]) / 2 << ", ";

	cout << "\nCNV vs LIN: \n";

	for (int i = 0; i < len / 2; i++)
		cout << tc3[i] << ", ";

	cout << endl;

	for (int i = 0; i < len / 2; i++)
		cout << tl3[i] << ", ";

	cout << "\nLIN1 vs LIN2: ";

	for (int i = 0; i < len / 2; i++)
		cout << (tl1[i] + tl2[i]) / 2 << ", ";

	cout << "\nLIN1 vs LIN3: ";

	for (int i = 0; i < len / 2; i++)
		cout << tl4[i] << ", ";

	cout << endl;

	cout << "\nLIN1 vs LIN4: ";

	for (int i = 0; i < len / 2; i++)
		cout << tl5[i] << ", ";

	cout << endl;

# elif WHICH == COMPLEMENT
	for (int i = 0; i < len / 2; i++)
	{
		auto ctx1 = Scheduler::selectContext(options[i]);
		ctx1->select(0);

		for (int j = 0; j < warmup; j++)
		{
			auto out = conv1->forward(inc1);
			out = conv2->forward(inc2);
			// out = lin2->forward(inl2);
			// out = lin3->forward(inl3);
			// out = lin4->forward(inl4);
		}

		auto ctx2 = Scheduler::selectContext(68 - options[i]);
		ctx2->select(0);

		for (int j = 0; j < warmup; j++)
		{
			auto out = conv1->forward(inc1);
			out = conv2->forward(inc2);
			// out = lin2->forward(inl2);
			// out = lin3->forward(inl3);
			// out = lin4->forward(inl4);
		}

		ctx1->release(0);
		ctx2->release(0);

		// int sec = 20 / options[i] + 2;
		int sec = 1;

		// speedup_double(sec, ctx1, conv1, inc1, "CNV1", tc1 + i, ctx2, conv2, inc2, "CNV2", tc2 + i);
		// speedup_double(sec, ctx1, conv1, inc1, "CNV", tc3 + i, ctx2, lin2, inl2, "LIN", tl3 + i);
		// speedup_double(sec, ctx2, conv1, inc1, "CNV", tc4 + i, ctx1, lin2, inl2, "LIN", tl4 + i);
		// speedup_double(sec, ctx1, lin1, inl1, "LIN1", tl1 + i, ctx2, lin2, inl2, "LIN2", tl2 + i);
		// speedup_double(sec, ctx1, lin1, inl1, "LIN1", tl5 + i, ctx2, lin3, inl3, "LIN3", tl9 + i);
		// speedup_double(sec, ctx2, lin1, inl1, "LIN1", tl6 + i, ctx1, lin3, inl3, "LIN3", tl9 + i);
		// speedup_double(sec, ctx1, lin1, inl1, "LIN1", tl7 + i, ctx2, lin4, inl4, "LIN4", tl9 + i);
		// speedup_double(sec, ctx2, lin1, inl1, "LIN1", tl8 + i, ctx1, lin4, inl4, "LIN4", tl9 + i);

		cout << endl;
	}

	cout << "CNV1 vs CNV2: \n";

	for (int i = 0; i < len / 2; i++)
		cout << tc1[i] << ", ";

	cout << endl;

	for (int i = len / 2 - 1; i >= 0; i--)
		cout << ", " << tc2[i];

	cout << "\nCNV vs LIN: \n";

	for (int i = 0; i < len / 2; i++)
		cout << tc3[i] << ", ";

	cout << endl;

	for (int i = len / 2 - 1; i >= 0; i--)
		cout << ", " << tc4[i];

	cout << endl;

	for (int i = 0; i < len / 2; i++)
		cout << tl4[i] << ", ";

	cout << endl;

	for (int i = len / 2 - 1; i >= 0; i--)
		cout << ", " << tl3[i];

	cout << "\nLIN1 vs LIN2: \n";

	for (int i = 0; i < len / 2; i++)
		cout << tl1[i] << ", ";

	cout << endl;

	for (int i = len / 2 - 1; i >= 0; i--)
		cout << ", " << tl2[i];

	cout << "\nLIN1 vs LIN3: \n";

	for (int i = 0; i < len / 2; i++)
		cout << tl5[i] << ", ";

	cout << endl;

	for (int i = len / 2 - 1; i >= 0; i--)
		cout << ", " << tl6[i];

	cout << "\nLIN1 vs LIN4: \n";

	for (int i = 0; i < len / 2; i++)
		cout << tl7[i] << ", ";

	cout << endl;

	for (int i = len / 2 - 1; i >= 0; i--)
		cout << ", " << tl8[i];

	cout << endl;
# endif
# endif
}

steady_clock::time_point tailing_start_thrd(MyContext* ctx, Sequential model, Tensor input)
{
	auto t = steady_clock::now();
	ctx->select(0);
	auto out = model->forward(input);
	ctx->release(0);
	return t;
}

steady_clock::time_point tailing_end_thrd(bool sync, MyContext* ctx, Sequential model, Tensor input)
{
	ctx->select(0);

	if (sync)
		cuCtxSynchronize();

	auto out = model->forward(input);
	ctx->release(0);
	cuCtxSynchronize();
	return steady_clock::now();
}

bool tailing_dummy_thrd(bool* stop, MyContext* ctx, Sequential module, Tensor input)
{
	ctx->select(0);

	while(!*stop)
		auto out = module->forward(input);

	ctx->release(0);

	return true;
}

void test_tailing(int argc, char** argv)
{
	string folder = "tail/";
	string fileNameSync = "tail_sync", fileNameAsync = "tail_async";
	int warmup = 100, repeat = 1000;

	printf("Starting process for \"Tailing Effect\" simulation with %s SMs:\n", argv[1]);

	auto conv1 = Sequential(Conv2d(Conv2dOptions(3, 3, 3).stride(1).padding(2)));
	conv1->eval();
	conv1->to(kCUDA);
	auto inc1 = torch::randn({3, 400, 400}, kCUDA);

	auto conv2 = Sequential(Conv2d(Conv2dOptions(3, 3, 3).stride(1).padding(2)));
	conv2->eval();
	conv2->to(kCUDA);
	auto inc2 = torch::randn({3, 400, 400}, kCUDA);

	auto conv3 = Sequential(Conv2d(Conv2dOptions(3, 3, 3).stride(1).padding(2)));
	conv3->eval();
	conv3->to(kCUDA);
	auto inc3 = torch::randn({3, 1024, 1024}, kCUDA);

	int inputSize1 = 512;
	auto lin1 = Sequential(Linear(inputSize1, inputSize1 / 2));
	lin1->eval();
	lin1->to(kCUDA);
	auto inl1 = torch::rand(inputSize1, kCUDA);

	int inputSize2 = 512;
	auto lin2 = Sequential(Linear(inputSize2, inputSize2 / 2));
	lin2->eval();
	lin2->to(kCUDA);
	auto inl2 = torch::rand(inputSize2, kCUDA);

	int inputSize3 = 512;
	auto lin3 = Sequential(Linear(inputSize2, inputSize3 / 2));
	lin3->eval();
	lin3->to(kCUDA);
	auto inl3 = torch::rand(inputSize3, kCUDA);

	int options[2] = {min(atoi(argv[1]), 68 - atoi(argv[1])), max(atoi(argv[1]), 68 - atoi(argv[1]))};
	Scheduler::initialize(options, 2);
	auto ctx = Scheduler::selectContext(atoi(argv[1]));
	auto ctxDummy = Scheduler::selectContext(68 - atoi(argv[1]));

	Sequential mod1 = conv1, mod2 = conv2, modDummy = lin3;
	Tensor in1 = inc1, in2 = inc2, inDummy = inl3;
	bool stop;
	stop = atoi(argv[1]) < 68;

	ctx->select(0);

	for (int i = 0; i < warmup; i++)
	{
		auto out = mod1->forward(in1);
		out = mod2->forward(in2);
		out = modDummy->forward(inDummy);
	}

	ctx->release(0);

	double resultAsync = 0, resultSync = 0;

	auto thDummy = async(tailing_dummy_thrd, &stop, ctxDummy, modDummy, inDummy);

	for (int i = 0; i < repeat; i++)
	{
		auto tStart = steady_clock::now();
		auto th1 = async(tailing_start_thrd, ctx, mod1, in1);
		usleep(200);
		auto th2 = async(tailing_end_thrd, false, ctx, mod2, in2);

		auto t1 = th1.get();
		auto t2 = th2.get();
		auto tEnd = steady_clock::now();
		duration<double> d = t2 - t1;
		// duration<double> d = tEnd - tStart;
		// resultAsync += d.count();

		tStart = steady_clock::now();
		th1 = async(tailing_start_thrd, ctx, mod1, in1);
		usleep(200);
		th2 = async(tailing_end_thrd, true, ctx, mod2, in2);

		t1 = th1.get();
		t2 = th2.get();
		tEnd = steady_clock::now();
		d = t2 - t1;
		// d = tEnd - tStart;
		resultSync += d.count();

		tStart = steady_clock::now();
		th1 = async(tailing_start_thrd, ctx, mod1, in1);
		usleep(200);
		th2 = async(tailing_end_thrd, false, ctx, mod2, in2);

		t1 = th1.get();
		t2 = th2.get();
		tEnd = steady_clock::now();
		d = t2 - t1;
		// d = tEnd - tStart;
		resultAsync += d.count();
	}

	stop = true;
	auto dummy = thDummy.get();

	resultAsync /= repeat;
	resultSync /= repeat;



	cout << fixed << setprecision(2);
	cout << "Result Wihout Synchronization:\t" << resultAsync * 1000 * 1000 << "us" << endl;
	cout << "Result With Synchronization:\t" << resultSync * 1000 * 1000 << "us" << endl;
	cout << "Overhead:\t\t\t" << showpos << (resultSync - resultAsync) / resultAsync * 100 << "%" << endl << endl;

	c10::cuda::CUDACachingAllocator::emptyCache();
}

void tabs(size_t num)
{
  for (size_t i = 0; i < num; i++)
    std::cout << "\t";
}

void print_modules(const shared_ptr<Module>& module, size_t level = 0)
{
	// module->pretty_print(cout);
	// cout << endl;
	for (const auto &param : module->modules())
	{
		param->pretty_print(cout);
		cout << endl;
	}
}

int main(int argc, char** argv)
{
	// test();
	// test_container();
	// test_speedup();
	// test_parallel();
	// test_interference();
	// test_speedup_double(argc, argv);
	// test_tailing(argc, argv);

	// cout << "ResNet18\n";
	// auto model = resnet18(1000);
	// print_modules(model);
	// cout << endl
	// 		 << endl;

	// cout << "ResNet34\n";
	// model = resnet34(1000);
	// print_modules(model);
	// cout << endl
	// 		 << endl;

	// cout << "ResNet50\n";
	// auto model2 = resnet50(1000);
	// print_modules(model2);
	// cout << endl
	// 		 << endl;

	// cout << "ResNet101\n";
	// model2 = resnet101(1000);
	// print_modules(model2);
	// cout << endl
	// 		 << endl;

	// cout << "ResNet152\n";
	// model2 = resnet152(1000);
	// print_modules(model2);
	// cout << endl
	// 		 << endl;

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
}