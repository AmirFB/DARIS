# ifndef __CONTAINER__
# define __CONTAINER__

# include <opr.hpp>
# include <ctxd.hpp>

# include <torch/torch.h>

# include <chrono>
# include <future>
# include <stdio.h>
# include <memory>

# include <spdlog/spdlog.h>
# include <spdlog/sinks/basic_file_sink.h>

using namespace torch;
using namespace torch::nn;

using namespace std;
using namespace chrono;
using namespace spdlog;

namespace FGPRS
{
	class Operation;

	class MyContainer : public Module
	{
	private:
		time_point<steady_clock> _lastArrival;

	public:
		bool highPriority = false;
		vector<vector<shared_ptr<Operation>>> operations;
		int interval;
		MyContext* currentContext;

		bool delayed = false;
		ModuleTracker tracker;
		string name;
		int bcet, wcet;
		shared_ptr<spdlog::logger> analyzeLogger, deadlineLogger, scheduleLogger;

		MyContainer() : Module() {}
		MyContainer(const MyContainer& container) : Module(container) {}

		void initLoggers();
		void clearAnalyzeLogger();
		void clearScheduleLogger();

		virtual Tensor forward(Tensor input) { return input; }
		virtual Tensor release(Tensor input);

		void analyze(int warmup, int repeat, Tensor input);
		virtual Tensor analyze(int warmup, int repeat, Tensor input);

		void assignDeadline();
		virtual void setAbsoluteDeadline(steady_clock::time_point start);
	};
}

# endif