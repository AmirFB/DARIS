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

# include <tests.hpp>
// # include <ctx.hpp>
# include <schd.hpp>

using namespace std;
using namespace chrono;
using namespace torch;
using namespace torch::nn;
using namespace FGPRS;

# define MODULE_COUNT	2

vector<double> repeat(int timer,
	MyContext* ctx1, Sequential mod1, Tensor in1,
	MyContext* ctx2, Sequential mod2, Tensor in2);
double run(int delay,
	MyContext* ctx1, Sequential mod1, Tensor in1,
	MyContext* ctx2, Sequential mod2, Tensor in2);
void thrd(MyContext* ctx, Sequential mod, Tensor in);

void testInterference(char** argv)
{
	NoGradGuard no_grad;
	Sequential cnv[MODULE_COUNT], lin[MODULE_COUNT];
	Tensor inc[MODULE_COUNT], inl[MODULE_COUNT];

	int smCount, timer;

	smCount = atoi(argv[0]);
	timer = (int)(atof(argv[1]));

	printf("Running \"Interference\" simulation. (SM count: %d)\n", smCount);

	for (int i = 0; i < MODULE_COUNT; i++)
	{
		cnv[i] = Sequential(Conv2d(Conv2dOptions(512, 1024, 3).stride(2).padding(1)));
		inc[i] = torch::randn({ 512, 48, 48 }, kCUDA);

		lin[i] = Sequential(Linear(2048, 100));
		inl[i] = torch::randn(2048, kCUDA);
	}

	for (int i = 0; i < MODULE_COUNT; i++)
	{
		cnv[i]->eval();
		cnv[i]->to(kCUDA);

		lin[i]->eval();
		lin[i]->to(kCUDA);
	}

	int options[] = { min(smCount, 68 - smCount), max(smCount, 68 - smCount) };

	if (!Scheduler::initialize(options, 2))
	{
		cout << "CUDA initialization failed.\n";
		return;
	}

	cout << "-------------------------------------------------------------------------------\n";
	cout << "Warming up\n";

	auto ctx1 = Scheduler::selectContext(smCount);
	ctx1->select();
	auto ctx2 = Scheduler::selectContext(68 - smCount);
	ctx1->release();

	Tensor dummy;
	steady_clock::time_point tstart, now, tend;
	tstart = steady_clock::now();
	tend = tstart + milliseconds(timer);

	auto ctxD = Scheduler::selectContext(68);
	ctxD->select();

	while (true)
	{
		for (int i = 0; i < MODULE_COUNT; i++)
		{
			ctxD->select();
			dummy = cnv[i]->forward(inc[i]);
			dummy = lin[i]->forward(inl[i]);
			cuCtxSynchronize();
			ctxD->release();

			ctx1->select();
			dummy = cnv[i]->forward(inc[i]);
			dummy = lin[i]->forward(inl[i]);
			cuCtxSynchronize();
			ctx1->release();

			ctx2->select();
			dummy = cnv[i]->forward(inc[i]);
			dummy = lin[i]->forward(inl[i]);
			cuCtxSynchronize();
			ctx2->release();
		}

		cuCtxSynchronize();
		now = steady_clock::now();

		if (tend <= steady_clock::now())
			break;
	}

	vector<double> results(0), temp(0);

	temp = repeat(timer,
		ctx1, cnv[0], inc[0],
		ctx2, lin[0], inl[0]);
	results.insert(results.end(), temp.begin(), temp.end());

	cout << "Saving results\n";
	writeToFile("interference", smCount, results);
	cout << "-------------------------------------------------------------------------------\n\n";
}

vector<double> repeat(int timer,
	MyContext* ctx1, Sequential mod1, Tensor in1,
	MyContext* ctx2, Sequential mod2, Tensor in2)
{
	vector<double> output(0);
	double result, bcet1, bcet2;
	steady_clock::time_point tstart, tend, now;
	static int round = 0;
	int delayLimit[3];
	int seed = time(NULL);
	int delay;

	printf("Round %d Results:\n", ++round);

	int count = 0;
	tstart = steady_clock::now();
	tend = tstart + milliseconds(timer);

	ctx1->select();

	while (true)
	{
		auto dummyOutput = mod1->forward(in1);
		cuCtxSynchronize();
		count++;
		now = steady_clock::now();

		if (tend <= steady_clock::now())
			break;
	}

	ctx1->release();

	duration<double> d = now - tstart;
	bcet1 = d.count() * 1000000 / count;

	count = 0;
	tstart = steady_clock::now();
	tend = tstart + milliseconds(timer);

	ctx2->select();

	while (true)
	{
		auto dummyOutput = mod2->forward(in2);
		cuCtxSynchronize();
		count++;
		now = steady_clock::now();

		if (tend <= steady_clock::now())
			break;
	}

	ctx2->release();

	d = now - tstart;
	bcet2 = d.count() * 1000000 / count;
	output.push_back(bcet2);

	delayLimit[0] = max(bcet1 - bcet2, 0.0) * 0.01;
	delayLimit[1] = max(bcet1 - bcet2, 0.0) * 0.5;
	delayLimit[2] = max(bcet1 - bcet2, 0.0) * 0.95;

	cout << "Limits: " << delayLimit[0] << "->" << delayLimit[1] << "->" << delayLimit[2] << endl;
	cout << "BCET: " << bcet2 << endl << endl;

	for (int i = 0; i < 4; i++)
	{
		count = 0;
		result = 0;

		if (i < 3)
			delay = delayLimit[i];

		else
			srand(seed);

		tstart = steady_clock::now();
		tend = tstart + milliseconds(timer);

		while (true)
		{
			if (i == 3)
				delay = delayLimit[0] + (float)rand() / RAND_MAX * (delayLimit[1] - delayLimit[0]);

			result += run(delay,
				ctx1, mod1, in1,
				ctx2, mod2, in2);

			count++;
			now = steady_clock::now();

			if (tend <= steady_clock::now())
				break;
		}

		result = result / count * 1000000;
		output.push_back(result);
		cout << result << endl;
	}

	cout << endl;
	return output;
}

double run(int delay,
	MyContext* ctx1, Sequential mod1, Tensor in1,
	MyContext* ctx2, Sequential mod2, Tensor in2)
{
	steady_clock::time_point t1, t2, t3;
	vector<double> output(2);
	duration<double> d;

	auto th1 = async(thrd, ctx1, mod1, in1);
	usleep(delay);

	t1 = steady_clock::now();
	thrd(ctx2, mod2, in2);
	t2 = steady_clock::now();

	th1.get();

	d = t2 - t1;
	return d.count();
}

void thrd(MyContext* ctx, Sequential mod, Tensor in)
{
	ctx->select();
	ctx->lock();
	auto out = mod->forward(in);
	ctx->unlock();
	ctx->release();
}