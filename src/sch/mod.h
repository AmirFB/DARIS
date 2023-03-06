# ifndef __CONTAINER__
# define __CONTAINER__

# include <ctxd.h>
# include <operation.h>

# include <torch/torch.h>

# include <chrono>
# include <future>
# include <stdio.h>
# include <memory>

using namespace torch;
using namespace torch::nn;

using namespace std;
using namespace std::chrono;

namespace FGPRS
{
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

		vector<shared_ptr<MyContainer>> _containers;
		int _maxLevel = 0;

		vector<vector<ContextData>> contextData
			= { vector<ContextData>(), vector<ContextData>(), vector<ContextData>() };

		MyContainer(): Module() {}
		MyContainer(const MyContainer& container): Module(container) {}

		virtual void assignOperations() {}
		vector<shared_ptr<Operation>> getOperations(int level);

		void copyOperations(string parentName, MyContainer& container, int level = 1);

		void addOperations(vector<Operation> operations);
		void addOperations(string parentName, vector<Operation> operations);

		void addOperations(vector<Operation> operations, int level);
		void addOperations(string parentName, vector<Operation> operations, int level);

		virtual Tensor forward(Tensor input) { return input; }

		void analyze(int warmup, int repeat, Tensor input);
		virtual Tensor analyze(int warmup, int repeat, Tensor input, int level);

		void assignExecutionTime(int contextIndex);
		virtual double assignExecutionTime(int level, int contextIndex, double executionTimetack);

		void assignDeadline(double quota, int contextIndex);
		virtual double assignDeadline(double quota, int level, int contextIndex, double deadlineStack);
		void setAbsoluteDeadline(int level, steady_clock::time_point start);

		template <typename ModuleType>
		shared_ptr<Operation> addOperation(string name, shared_ptr<ModuleType> module, int level = 0)
		{
			auto operation = make_shared<Operation>(name, module);

			if (level == 0 || level == 1)
				operations[0].push_back(operation);

			if (level == 0 || level == 2)
				operations[1].push_back(operation);

			if (level == 0 || level == 3)
				operations[2].push_back(operation);

			return operation;
		}
	};

	class MySequential: public MyContainer, public ModuleHolder<SequentialImpl>
	{
	public:
		using ModuleHolder<SequentialImpl>::ModuleHolder;

		MySequential(): ModuleHolder() {}
		MySequential(initializer_list<NamedAnyModule> named_modules): ModuleHolder(make_shared<SequentialImpl>(move(named_modules))) {}

		void setLevel(int level) { _maxLevel = level; }

		void addContainer(shared_ptr<MyContainer> container)
		{
			_containers.push_back(container);
		}

		void copyOperations(string parentName, MyContainer& container, int level = 1)
		{
			MyContainer::copyOperations(parentName, container, level);

			_maxLevel = container._maxLevel;
			_containers.insert(_containers.end(), container._containers.begin(), container._containers.end());
		}

		Tensor analyze(int warmup, int repeat, Tensor input, int level) override
		{
			if (level > _maxLevel || _containers.size() == 0)
				return MyContainer::analyze(warmup, repeat, input, level);

			for (auto cont : _containers)
				input = cont->analyze(warmup, repeat, input, level);

			return input;
		}

		double assignExecutionTime(int level, int contextIndex, double executionTimeStack)
		{
			double tempStack = 0, elapsedTime = 0, tempElapsed;

			if (level > _maxLevel || _containers.size() == 0)
				return MyContainer::assignExecutionTime(level, contextIndex, executionTimeStack);

			for (auto cont : _containers)
			{
				tempStack = cont->assignExecutionTime(level, contextIndex, executionTimeStack);
				tempElapsed = tempStack - executionTimeStack;
				elapsedTime += tempElapsed;
				executionTimeStack = tempStack;
			}

			level--;
			regulatedExecutionTime[level] = elapsedTime;
			return executionTimeStack;
		}

		double assignDeadline(double quota, int level, int contextIndex, double deadlineStack) override
		{
			if (level > _maxLevel || _containers.size() == 0)
				return MyContainer::assignDeadline(quota, level, contextIndex, deadlineStack);

			level--;

			for (auto cont : _containers)
				deadlineStack = cont->assignDeadline((cont->regulatedExecutionTime[level] / regulatedExecutionTime[level]) * quota, level + 1, contextIndex, deadlineStack);

			return deadlineStack;
		}
	};
}

# endif