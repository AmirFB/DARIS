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

# define MODULE_COUNT	3

vector<double> repeat(int timer,
	MyContext* ctxD1, Sequential modD1, Tensor inD1,
	MyContext* ctxD2, Sequential modD2, Tensor inD2,
	MyContext* ctx1, Sequential mod1, Tensor in1,
	MyContext* ctx2, Sequential mod2, Tensor in2);
vector<double> run(int sync, bool dummy, int delay,
	MyContext* ctxD, Sequential modD, Tensor inD,
	MyContext* ctx1, Sequential mod1, Tensor in1,
	MyContext* ctx2, Sequential mod2, Tensor in2);
void dummy_thrd(bool* stop, MyContext* ctx, Sequential mod, Tensor in);
void main_thrd(MyContext* ctx, Sequential mod, Tensor in);
void tail_thrd(int sync, MyContext* ctx, Sequential mod, Tensor in);

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
		inc[i] = torch::randn({ 512, 48, 48 }, kCUDA);

		lin[i] = Sequential(Linear(4096 * 4, 10000));
		inl[i] = torch::randn(4096 * 4, kCUDA);
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

	auto ctx = Scheduler::selectContext(smCount);
	ctx->select();
	auto ctxP = Scheduler::selectContext(68 - smCount);
	ctx->release();

	Tensor dummy;
	steady_clock::time_point tstart, now, tend;
	tstart = steady_clock::now();
	tend = tstart + milliseconds(5000);

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

			ctx->select();
			dummy = cnv[i]->forward(inc[i]);
			dummy = lin[i]->forward(inl[i]);
			cuCtxSynchronize();
			ctx->release();

			ctxP->select();
			dummy = cnv[i]->forward(inc[i]);
			dummy = lin[i]->forward(inl[i]);
			cuCtxSynchronize();
			ctxP->release();
		}

		cuCtxSynchronize();
		now = steady_clock::now();

		if (tend <= steady_clock::now())
			break;
	}

	vector<double> results(0), temp(0);

	temp = repeat(timer,
		ctxP, cnv[2], inc[2],
		ctxP, lin[2], inl[2],
		ctx, cnv[0], inc[0],
		ctx, cnv[1], inc[1]);
	results.insert(results.end(), temp.begin(), temp.end());

	temp = repeat(timer,
		ctxP, cnv[2], inc[2],
		ctxP, lin[2], inl[2],
		ctx, cnv[0], inc[0],
		ctx, lin[1], inl[1]);
	results.insert(results.end(), temp.begin(), temp.end());

	temp = repeat(timer,
		ctxP, cnv[2], inc[2],
		ctxP, lin[2], inl[2],
		ctx, lin[0], inl[0],
		ctx, cnv[1], inc[1]);
	results.insert(results.end(), temp.begin(), temp.end());

	temp = repeat(timer,
		ctxP, cnv[2], inc[2],
		ctxP, lin[2], inl[2],
		ctx, lin[0], inl[0],
		ctx, lin[1], inl[1]);
	results.insert(results.end(), temp.begin(), temp.end());

	cout << "Saving results\n";
	writeToFile("tailing", smCount, results);
	cout << "-------------------------------------------------------------------------------\n\n";
}

vector<double> repeat(int timer,
	MyContext* ctxD1, Sequential modD1, Tensor inD1,
	MyContext* ctxD2, Sequential modD2, Tensor inD2,
	MyContext* ctx1, Sequential mod1, Tensor in1,
	MyContext* ctx2, Sequential mod2, Tensor in2)
{
	vector<double> temp(2), result(2), output(0);
	steady_clock::time_point tstart, tend, now;
	static int round = 0;
	int delayLimit[2];
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
	delayLimit[0] = d.count() * 1000000 / count * 0.01;
	delayLimit[1] = d.count() * 1000000 / count * 0.95;

	output.push_back(d.count() * 1000000 / count);

	cout << "Limits: " << delayLimit[0] << "->" << delayLimit[1] << endl << endl;

	for (int dummy = 0; dummy < 3; dummy++)
	{
		for (int sync = 0; sync < 3; sync++)
		{
			for (int i = 0; i < 3; i++)
			{
				count = 0;
				result[0] = 0;
				result[1] = 0;

				if (i < 2)
					delay = delayLimit[i];

				else
					srand(seed);

				tstart = steady_clock::now();
				tend = tstart + milliseconds(sync != 1 ? timer : timer / 1000);

				while (true)
				{
					if (i == 2)
						delay = delayLimit[0] + (float)rand() / RAND_MAX * (delayLimit[1] - delayLimit[0]);

					if (dummy < 2)
					{
						temp = run(sync, dummy, delay,
							ctxD1, modD1, inD1,
							ctx1, mod1, in1,
							ctx2, mod2, in2);
					}

					else
					{
						temp = run(sync, dummy, delay,
							ctxD2, modD2, inD2,
							ctx1, mod1, in1,
							ctx2, mod2, in2);
					}

					cuCtxSynchronize();
					count++;
					result[0] += temp[0];
					result[1] += temp[1];
					now = steady_clock::now();

					if (tend <= steady_clock::now())
						break;
				}

				result[0] = result[0] / count * 1000000;
				result[1] = result[1] / count * 1000000;
				output.insert(output.end(), result.begin(), result.end());
				cout << result[0] << ", " << result[1] << endl;
			}

			cout << "--\n";
		}

		cout << "----\n";
	}

	cout << endl;
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

	if (sync == 1)
	{
		output[0] = 0;
		output[1] = 0;
		return output;
	}

	bool stop = (!dummy && ctxD->smCount > 1), finished = false;
	auto thD = async(dummy_thrd, &stop, ctxD, modD, inD);

	t1 = steady_clock::now();
	auto th1 = async(main_thrd, ctx1, mod1, in1);
	usleep(delay);
	auto th2 = async(tail_thrd, sync, ctx2, mod2, in2);

	th1.get();
	t2 = steady_clock::now();
	th2.get();
	t3 = steady_clock::now();
	stop = true;
	thD.get();
	d = t2 - t1;
	output[0] = d.count();
	d = t3 - t1;
	output[1] = d.count();

	return output;
}

void dummy_thrd(bool* stop, MyContext* ctx, Sequential mod, Tensor in)
{
	ctx->select();

	while (!*stop)
		auto out = mod->forward(in);

	ctx->release();
}

void main_thrd(MyContext* ctx, Sequential mod, Tensor in)
{
	ctx->select();
	ctx->lock();
	auto out = mod->forward(in);
	ctx->unlock();
	ctx->release();
}

void tail_thrd(int sync, MyContext* ctx, Sequential mod, Tensor in)
{
	ctx->select();

	if (sync == 1)
		// cuCtxSynchronize();
		return;

	else if (sync == 2)
		ctx->lock();

	auto out = mod->forward(in);

	if (sync == 2)
		ctx->unlock();

	ctx->release();
}