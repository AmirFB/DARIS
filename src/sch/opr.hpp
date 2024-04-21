# pragma once

# include <cnt.hpp>
# include <str.hpp>
# include <ctx.hpp>

# include <torch/torch.h>
# include <c10/cuda/CUDAStream.h>


# include <chrono>
# include <future>
# include <stdio.h>
# include <memory>

using namespace torch;
using namespace torch::nn;

using namespace std;
using namespace chrono;

namespace FGPRS
{
	class MyContainer;
	class MyModule;
	class MyStream;

	class Operation : public enable_shared_from_this<Operation>
	{
	private:
		Tensor* _input;
		bool _isNL = false;
		int _parallelCount = 0;

	public:
		int id;
		string name, fullName;
		shared_ptr<MyContainer> container;
		shared_ptr<SequentialImpl> sequential;
		shared_ptr<MyModule> module;
		int relativeDeadline, stackedDeadline;
		steady_clock::time_point absoluteDeadline;
		steady_clock::time_point startTime, finishTime;
		bool highPriority = false, isLast = false;
		int bcet, wcet, wret;
		bool priorDelayed = false, finished = false, delayed;
		bool released = false;
		MyContext* currentContext;
		MyStream* currentStream;

		Operation() {}
		Operation(string name, shared_ptr<MyContainer> container, shared_ptr<SequentialImpl> module, bool isLast);
		Operation(string name, shared_ptr<MyContainer> container, shared_ptr<MyModule> module, bool isLast, int parallelCount);

		static bool EDF(const Operation& op1, const Operation& op2)
		{
			return op1.absoluteDeadline < op2.absoluteDeadline;
		}

		static bool EDF(const shared_ptr<Operation>& op1, const shared_ptr<Operation>& op2)
		{
			return op1->absoluteDeadline < op2->absoluteDeadline;
		}

		void setAbsoluteDeadline(steady_clock::time_point start);
		Tensor forward(Tensor input);
		Tensor forward(Tensor input, MyContext* ctx, MyStream* str);
		Tensor analyze(int warmup, int repeat, Tensor input, int* timer);
		Tensor releaseSync(Tensor input);
		future<Tensor> releaseAsync(Tensor input);
	};
}