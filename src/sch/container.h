# ifndef __CONTAINER__
# define __CONTAINER__

# include <ctx.h>

# include <torch/torch.h>

# include <chrono>
# include <future>

using namespace torch;
using namespace torch::nn;

using namespace std;
using namespace std::chrono;

namespace FGPRS
{
	struct Container : Module
	{
		private:
		time_point<steady_clock> _lastArrival;
		future<Tensor> _ath;
		thread pThread;

		public:
		double* executionTime;
		int interval;

		Container();
		~Container();
		void start(Tensor *input, MyContext* ctx);
		Tensor getResult();
		virtual Tensor forward(Tensor input);
		virtual void initialize();
		virtual void analyze(Tensor* dummyInput, int warmup, int repeat);
		void run(Tensor *input, int repeat);
		void join();
	};
}

# endif