#ifndef __CONTAINER__
#define __CONTAINER__

#include <ctx.h>

#include <torch/torch.h>

#include <chrono>
#include <future>

#include <stdio.h>

using namespace torch;
using namespace torch::nn;

using namespace std;
using namespace std::chrono;

namespace FGPRS
{
	class Operation;

	class MyModule : public Module
	{
	private:
		time_point<steady_clock> _lastArrival;
		vector<Operation> _operations;

	public:
		double *executionTime;
		int interval;

		MyModule() : Module() {}
		MyModule(const MyModule &container) : Module(container) {}

		Tensor forward(Tensor);
		Tensor analyze(int warmup, int repeat, Tensor dummyData, vector<int> smOptions);

		template <typename ModuleType>
		void addOperation(string name, shared_ptr<ModuleType> module)
		{
			auto operation = Operation(name, module);
			_operations.push_back(operation);
		}

		virtual void assignOperations() {}
		vector<Operation> getOperations();
		void addOperations(vector<Operation> operations);
		void addOperations(string parentName, vector<Operation> operations);
	};

	class MySequential : public MyModule, public ModuleHolder<SequentialImpl>
	{
	public:
		using ModuleHolder<SequentialImpl>::ModuleHolder;

		MySequential() : ModuleHolder() {}
		MySequential(initializer_list<NamedAnyModule> named_modules) : ModuleHolder(make_shared<SequentialImpl>(move(named_modules))) {}
	};

	class Operation
	{
	private:
		string _name;
		string _fullName;
		Sequential _sequential;
		shared_ptr<Tensor> _output;
		vector<ContextData> _contextData;
		double _isolatedScalability, _occupiedScalability, _predictability;
		double _relativeDeadline;
		double _absoulteDeadline;

	public:
		Operation() {}
		string getName() { return _name; }
		void setName(string name)
		{
			_name = name;
			_fullName = name;
		}
		void setParentName(string parentName) { _fullName = parentName + "->" + _fullName; }

		template <typename ModuleType>
		Operation(string name, shared_ptr<ModuleType> module)
		{
			_name = name;
			_fullName = name;
			_sequential = Sequential(module);
		}

		Tensor analyze(int warmup, int repeat, Tensor input, vector<int> smOptions);
		bool isBasic();
		bool isComplex();
		bool isSequential();
		void assign();
	};
}

#endif