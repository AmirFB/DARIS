# ifndef __CONTAINER__
# define __CONTAINER__

# include <ctx.h>
# include <operation.h>

# include <torch/torch.h>

# include <chrono>
# include <future>

using namespace torch;
using namespace torch::nn;

using namespace std;
using namespace std::chrono;

namespace FGPRS
{

template <typename T> std::string type_name();
	struct Container : public Module
	{
		private:
		time_point<steady_clock> _lastArrival;
		vector<Operation> _operations;

		public:
		double* executionTime;
		int interval;

		Container() : Module(){}
		Container(const Container& container) : Module(container){}
		~Container() {}

		template<typename ModuleType>
		shared_ptr<ModuleType> register_module(
			string name,
			ModuleHolder<ModuleType> module_holder)
		{
			auto output = Module::register_module(name, module_holder);
			// auto operation = Operation<ModuleType>(name, output);
			auto operation = Operation(name, output);
			_operations.push_back(operation);
			// _operations[0] =  
			return output;
		}

		Tensor analyze(Tensor dummyData, vector<int> smOptions)
		{
			for (auto op : _operations)
			{
				// cout << "OK!\n";
				// cout << op.analyze(5, 20, torch::randn({3, 224, 224}, kCUDA)) << endl;
				// cout << op.getName() << endl;
				// cout << op.getName() << endl;

				if (op.getName() == "fc")
					dummyData = flatten(dummyData, 1);

				dummyData = op.analyze(25, 100, dummyData, smOptions);
				// cout << typeid(op).name() << endl;
				// cout << typeid(op).name() << endl;
				// cout << type_name<decltype(op)>() << endl;
				// cout << type_name<decltype(*op)>() << endl;
				// cout << type_name<decltype((op))>() << endl;
				// cout << type_name<decltype((*op))>() << endl;
			}

			return dummyData;
		}
	};
}

# endif