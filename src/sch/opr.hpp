# ifndef __OPERATION__
# define __OPERATION__

# include <cnt.hpp>
# include <ctxd.hpp>

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

	class Operation
	{
	private:
		Tensor* _input;

	public:
		int id;
		string name, fullName;
		shared_ptr<MyContainer> container;
		shared_ptr<Sequential> sequential;
		int relativeDeadline[3], stackedDeadline[3];
		steady_clock::time_point absoluteDeadline;
		steady_clock::time_point startTime, finishTime;
		bool highPriority = false, isLatest = false;
		int bcet, wcet;

		Operation() {}

		Operation(int id, string name, shared_ptr<MyContainer> container, shared_ptr<Sequential> module, bool highPriority, bool isLatest)
			: id(id), name(name), container(container), fullName(container->name + "->" + name),
			sequential(module), highPriority(highPriority), isLatest(isLatest)
		{
		}

		Tensor analyze(int warmup, int repeat, Tensor input, int index);
		Tensor releaseSync(Tensor input);
		Tensor releaseAsync(Tensor input);
		void setAbsoluteDeadline(steady_clock::time_point start);
	};
}

# endif