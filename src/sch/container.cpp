# include <container.h>

# include <schd.h>
# include <ctx.h>

# include <iostream>
# include <chrono>
# include <thread>
# include <torch/torch.h>

using namespace torch;
using namespace torch::nn;

using namespace FGPRS;

using namespace std;
using namespace std::chrono;

Container::Container() : Module(){}
Container::~Container() {}
Tensor Container::forward(Tensor input) { return torch::rand(1, kCUDA); }

void Container::start(Tensor *input, MyContext* ctx)
{
	_ath = async(launch::async, [this, input, ctx]()
	{
		ctx->select(executionTime[ctx->smCount]);
		auto output = forward(*input);
		ctx->release(executionTime[ctx->smCount]);
		return output;
	});

	// return torch::rand(1, kCUDA);
}

Tensor Container::getResult()
{
	return _ath.get();
	// return torch::rand(1, kCUDA);
}

void Container::initialize()
{
	cout << "You got me!\n";
}

void Container::analyze(Tensor* dummyInput, int warmup, int repeat)
{
	executionTime = (double*)malloc(sizeof(double) * (Scheduler::maxSmCount * 1));

	for (int i = 0; i < Scheduler::poolSize; i++)
	{
		auto ctx = Scheduler::selectContextByIndex(i);
		int smCount = ctx->smCount;

		ctx->select(0);

		for (int j = 0; j < warmup; j++)
		{
			start(dummyInput, ctx);
			auto output = getResult();
			cuCtxSynchronize();
		}

		auto t1 = steady_clock::now();

		for (int j = 0; j < repeat; j++)
		{
			start(dummyInput, ctx);
			auto output = getResult();
			cuCtxSynchronize();
		}

		ctx->release(0);
		auto t2 = steady_clock::now();

		duration<double> d = t2 - t1;
		executionTime[smCount] = d.count() / repeat;
	}
}

void Container::run(Tensor* input, int repeat)
{
	cout << "Interval: " << interval << endl;
	// pThread = (thread*)malloc(sizeof(thread));

	//*pThread
	pThread = thread([this, input, repeat]()
  { 
		int counter = repeat;
		auto deadline = steady_clock::now();

		while (counter--)
    { 
      deadline = deadline + milliseconds(interval);
      auto ctx = Scheduler::selectContextByQueueLoad(executionTime);
			start(input, ctx);
			auto out = getResult();
			this_thread::sleep_until(deadline);
		}
  });

	// pThread->join();
	// pThread.join();
	// return pThread;
	// return NULL;
}

void Container::join()
{
	pThread.join();
}