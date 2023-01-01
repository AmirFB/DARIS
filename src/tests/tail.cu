# include <stdio.h>
# include <stdlib.h>
# include <sys/stat.h>
# include <unistd.h>
# include <future>

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

# include "tests.h"
# include "ctx.h"
# include "schd.h"

using namespace std;
using namespace std::chrono;
using namespace torch;
using namespace torch::nn;
using namespace FGPRS;

# define MODULE_COUNT	3

vector<double> repeat(int timer, int delay,
	MyContext* ctxD, Sequential modD, Tensor inD,
	MyContext* ctx1, Sequential mod1, Tensor in1,
	MyContext* ctx2, Sequential mod2, Tensor in2);
vector<double> run(int sync, bool dummy, int delay,
	MyContext* ctxD, Sequential modD, Tensor inD,
	MyContext* ctx1, Sequential mod1, Tensor in1,
	MyContext* ctx2, Sequential mod2, Tensor in2);
void dummy_thrd(bool* stop, MyContext* ctx, Sequential mod, Tensor in);
void main_thrd(MyContext* ctx, Sequential mod, Tensor in, bool* finished);
void tail_thrd(int sync, MyContext* ctx, Sequential mod, Tensor in, bool* finished);

void testTailing(char** argv)
{
	NoGradGuard no_grad;
	Sequential cnv[MODULE_COUNT], lin[MODULE_COUNT];
	Tensor inc[MODULE_COUNT], inl[MODULE_COUNT];

	int smCount, timer;

	smCount = atoi(argv[0]);
	timer = (int)(atof(argv[1]) * 1000);

	printf("Running \"Tailing\" simulation. (SM count: %d)\n", smCount);

	for (int i = 0; i < MODULE_COUNT; i++)
	{
		cnv[i] = Sequential(Conv2d(Conv2dOptions(512, 1024, 3).stride(2).padding(1)));
		inc[i] = torch::randn({512, 48, 48}, kCUDA);

		lin[i] = Sequential(Linear(4096*4, 10000));
		inl[i] = torch::randn(4096*4, kCUDA);
	}

	for (int i = 0; i < MODULE_COUNT; i++)
	{
		cnv[i]->eval();
		cnv[i]->to(kCUDA);
		
		lin[i]->eval();
		lin[i]->to(kCUDA);
	}

	int options[] = {min(smCount, 68 - smCount), max(smCount, 68 - smCount)};
	
	if (!Scheduler::initialize(options, 1))
	{
		cout << "CUDA initialization failed.\n";
		return;
	}

	cout << "-------------------------------------------------------------------------------\n";
	cout << "Warming up\n";

	auto ctx = Scheduler::selectContext(smCount);
	ctx->select(0);
	auto ctxP = Scheduler::selectContext(68 - smCount);
	ctx->release(0);

	Tensor dummy;
	steady_clock::time_point tstart, now, tend;
	tstart = steady_clock::now();
	tend = tstart + milliseconds(timer);

	auto ctxD = Scheduler::selectContext(68);
	ctxD->select(0);

	while (true)
	{
		for (int i = 0; i < MODULE_COUNT; i++)
		{
			ctxD->select(0);
			dummy = cnv[i]->forward(inc[i]);
			dummy = lin[i]->forward(inl[i]);
			cuCtxSynchronize();
			ctxD->release(0);
			
			ctx->select(0);
			dummy = cnv[i]->forward(inc[i]);
			dummy = lin[i]->forward(inl[i]);
			cuCtxSynchronize();
			ctx->release(0);
			
			ctxP->select(0);
			dummy = cnv[i]->forward(inc[i]);
			dummy = lin[i]->forward(inl[i]);
			cuCtxSynchronize();
			ctxP->release(0);
		}
		
		cuCtxSynchronize();
		now = steady_clock::now();

		if (tend <= steady_clock::now())
			break;
	}
	
	ctxD->release(0);
	ctx->select(0);

	while (true)
	{
		for (int i = 0; i < MODULE_COUNT; i++)
		{
			dummy = cnv[i]->forward(inc[i]);
			dummy = lin[i]->forward(inl[i]);
		}
		
		cuCtxSynchronize();
		now = steady_clock::now();

		if (tend <= steady_clock::now())
			break;
	}
	
	ctx->release(0);
	
	steady_clock::time_point t1, t2;
	duration<double> d;
	vector<double> results(MODULE_COUNT);

	
	
	cout << "Saving results\n";
	writeToFile("speedup", smCount, results);
	cout << "-------------------------------------------------------------------------------\n\n";
}

vector<double> repeat(int timer, int delay,
	MyContext* ctxD, Sequential modD, Tensor inD,
	MyContext* ctx1, Sequential mod1, Tensor in1,
	MyContext* ctx2, Sequential mod2, Tensor in2)
{
	vector<double> temp(2), result(2), output(0);
	steady_clock::time_point tstart, tend, now;

	for (int sync = 0; sync < 3; sync++)
	{
		for (int dummy = 0; dummy < 3; dummy++)
		{
			int count = 0;
			result[0] = 0;
			result[1] = 0;

			tstart = steady_clock::now();
			tend = tstart + milliseconds(timer);

			while (true)
			{
				temp = run(sync, dummy, delay,
					ctxD, modD, inD,
					ctx1, mod1, in1,
					ctx2, mod2, in2);
				cuCtxSynchronize();
				count++;
				result[0] += temp[0];
				result[1] += temp[1];
				now = steady_clock::now();

				if (tend <= steady_clock::now())
					break;
			}

			output.insert(output.end(), result.begin(), result.end());
		}
	}

	return output;
}

vector<double> run(int sync, bool dummy, int delay,
	MyContext* ctxD, Sequential modD, Tensor inD,
	MyContext* ctx1, Sequential mod1, Tensor in1,
	MyContext* ctx2, Sequential mod2, Tensor in2)
{
	steady_clock::time_point t1, t2, t3;
	vector<double> output(2);
	duration<double> d;
	bool stop = dummy, finished = false;

	auto thD = async(dummy_thrd, &stop, ctxD, modD, inD);
	
	t1 = steady_clock::now();
	auto th1 = async(main_thrd, ctx1, mod1, in1, &finished);
	usleep(delay);
	auto th2 = async(tail_thrd, sync, ctx2, mod2, in2, &finished);
	
	th1.get();
	t2 = steady_clock::now();
	th2.get();
	t3 = steady_clock::now();
	d = t2 - t1;
	output[0] = d.count();
	d = t3 - t1;
	output[1] = d.count();

	return output;
}

void dummy_thrd(bool* stop, MyContext* ctx, Sequential mod, Tensor in)
{
	ctx->select(0);

	while(!*stop)
		auto out = mod->forward(in);

	ctx->release(0);
}

void main_thrd(MyContext* ctx, Sequential mod, Tensor in, bool* finished)
{
	ctx->select(0);
	auto out = mod->forward(in);
	*finished = true;
	ctx->release(0);
}

void tail_thrd(int sync, MyContext* ctx, Sequential mod, Tensor in, bool* finished)
{
	ctx->select(0);

	if (sync == 1)
		cuCtxSynchronize();
	
	else if (sync == 2)
		while (!*finished);

	auto out = mod->forward(in);
	ctx->release(0);
	cuCtxSynchronize();
}