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

	class MyContainer: public Module
	{
	private:
		time_point<steady_clock> _lastArrival;

	public:
		vector<vector<shared_ptr<Operation>>> operations
			= { vector<shared_ptr<Operation>>() , vector<shared_ptr<Operation>>() , vector<shared_ptr<Operation>>() };
		double* executionTime;
		int interval;
		double deadlineQuota, regulatedExecutionTime[3];

		vector<shared_ptr<MyContainer>> containers;
		int _maxLevel = 0;
		string _name;

		shared_ptr<spdlog::logger> analyzeLogger, deadlineLogger, scheduleLogger;

		vector<vector<ContextData>> contextData
			= { vector<ContextData>(), vector<ContextData>(), vector<ContextData>() };

		MyContainer(): Module() {}
		MyContainer(const MyContainer& container): Module(container) {}

		void initLoggers(string name);
		void clearScheduleLogger(string name);

		virtual void assignOperations() {}
		vector<shared_ptr<Operation>> getOperations(int level);

		void copyOperations(string parentName, MyContainer& container, int level = 1);

		void addOperations(vector<Operation> operations);
		void addOperations(string parentName, vector<Operation> operations);

		void addOperations(vector<Operation> operations, int level);
		void addOperations(string parentName, vector<Operation> operations, int level);

		virtual Tensor forward(Tensor input) { return input; }
		virtual Tensor schedule(string name, Tensor input, int level);

		void analyze(int warmup, int repeat, Tensor input);
		virtual Tensor analyze(int warmup, int repeat, Tensor input, int level);

		virtual double assignExecutionTime(int level, int contextIndex, double executionTimetack);

		void assignDeadline(double quota, int contextIndex);
		virtual double assignDeadline(double quota, int level, int contextIndex, double deadlineStack);
		void setAbsoluteDeadline(int level, steady_clock::time_point start);

		template <typename ModuleType>
		shared_ptr<Operation> addOperation(MyContainer* owner, string name, shared_ptr<ModuleType> module, int level = 0)
		{
			auto operation = make_shared<Operation>(owner, name, module);

			if (level == 0 || level == 1)
				operations[0].push_back(operation);

			if (level == 0 || level == 2)
				operations[1].push_back(operation);

			if (level == 0 || level == 3)
				operations[2].push_back(operation);

			return operation;
		}
	};
}

# endif