# ifndef __OPERATION__
# define __OPERATION__

# include <ctxd.hpp>

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
	class Operation
	{
	private:
		string _name, _fullName, _lastParentName;
		thread _th;
		Tensor* _output;
		MyContext* _chosenContext;
		bool _isException;

		static double exceptionThreshold;

	public:
		Sequential sequential;
		double relativeDeadline[3], stackedDeadline[3];
		steady_clock::time_point absoluteDeadline;
		double isolatedScalability, occupiedScalability, predictability;
		vector<ContextData> contextData;
		steady_clock::time_point startTime;

		Operation() {}
		string getName();
		string getFullName();
		void setName(string name);
		void setParentName(string parentName);

		template <typename ModuleType>
		Operation(string name, shared_ptr<ModuleType> module)
		{
			_name = name;
			_fullName = name;
			sequential = Sequential(module);
		}

		Tensor analyze(int warmup, int repeat, Tensor input);

		void start(Tensor input);
		Tensor getResult();
		Tensor runSync(Tensor input);

		void startSchedule(string name, Tensor input);
		Tensor scheduleSync(string name, Tensor input);

		double getRegulatedExecutionTime(int contextIndex);
		void setAbsoluteDeadline(int level, steady_clock::time_point start);
	};
}

# endif