# ifndef __OPERATION__
# define __OPERATION__

# include <schd.h>
# include <ctx.h>

# include <unistd.h>
# include <math.h>

# include <torch/torch.h>
# include <torch/script.h>
# include <c10/cuda/CUDAStream.h>
# include <ATen/cuda/CUDAContext.h>

# include <c10/cuda/CUDACachingAllocator.h>

using namespace std;
using namespace std::chrono;
using namespace torch;
using namespace torch::nn;
using namespace FGPRS;

namespace FGPRS
{
	class Operation
	{
		private:
		string _name;
		Sequential _sequential;
		shared_ptr<Tensor> _output;
		vector<ContextData> _contextData;
		double _isolatedScalability, _occupiedScalability, _predictability;
		double _relativeDeadline;
		double _absoulteDeadline;

		public:
		string getName() { return _name; }
		void setName(string name) { _name = name; }
		Operation() {}

		template<typename ModuleType>
		Operation(string name, shared_ptr<ModuleType> module)
		{
			_name = name;
			_sequential = Sequential(module);
		}

		Tensor analyze(int warmup, int repeat, Tensor input, vector<int> smOptions)
		{
			Tensor output;
			bool first = true;
			steady_clock::time_point t1, t2, tStart, tEnd, tNow;
			int countIsolated, countOccupied;

			_predictability = 0;
			_isolatedScalability = 0;
			_occupiedScalability = 0;

			cout << _name << ":" << endl;

			Scheduler::selectDefaultContext();
			_contextData.clear();



			tStart = steady_clock::now();
			tEnd = tStart + milliseconds(warmup);

			while (true)
			{
				output = _sequential->forward(input);

				if (tEnd <= steady_clock::now())
					break;
			}

			for(auto sm : smOptions)
			{
				auto ctx = Scheduler::selectContext(sm);
				ctx.select();

				tStart = steady_clock::now();
				tEnd = tStart + milliseconds(warmup);

				while (true)
				{
					output = _sequential->forward(input);

					if (tEnd <= steady_clock::now())
						break;
				}

				countIsolated = 0;
				tStart = steady_clock::now();
				tEnd = tStart + milliseconds(repeat);

				while (true)
				{
					output = _sequential->forward(input);
					countIsolated++;
					tNow = steady_clock::now();

					if (tEnd <= tNow)
						break;
				}

				duration<double> d1 = tNow - tStart;

				ctx.release();
				auto doom = Scheduler::startDummy(68 - sm);
				usleep(10000);
				ctx.select();
				
				countOccupied = 0;
				tStart = steady_clock::now();
				tEnd = tStart + milliseconds(repeat);

				while (true)
				{
					output = _sequential->forward(input);
					countOccupied++;
					tNow = steady_clock::now();

					if (tEnd <= tNow)
						break;
				}

				duration<double> d2 = tNow - tStart;

				Scheduler::stopDummy();

				ctx.release();

				_contextData.push_back(ContextData(ctx, d1.count() / countIsolated * 1000000, d2.count() / countOccupied * 1000000));
				cout << "\t" << ctx.smCount << "\t" << _contextData.back().isolatedExecutionTime << "us" << ", " << _contextData.back().occupiedExecutionTime << "us";

				_predictability += 1 - (_contextData.back().occupiedExecutionTime - _contextData.back().isolatedExecutionTime) / _contextData.back().occupiedExecutionTime;

				if (first)
				{
					first = false;
					continue;
				}

				double desired, isolatedGain, occupiedGain;

				desired = (double)_contextData.back().smCount / _contextData.end()[-2].smCount;
				isolatedGain = _contextData.end()[-2].isolatedExecutionTime / _contextData.back().isolatedExecutionTime;
				occupiedGain = _contextData.end()[-2].occupiedExecutionTime / _contextData.back().occupiedExecutionTime;
				
				_isolatedScalability += max((isolatedGain - 1) / (desired - 1), 0.0);
				_occupiedScalability += max((occupiedGain - 1) / (desired - 1), 0.0);
			}

			_predictability /= 4;
			_isolatedScalability /= 3;
			_occupiedScalability /= 3;

			cout << endl << "Params: " << _predictability << "\t" << _isolatedScalability << "\t" << _occupiedScalability << endl << endl;

			return output;
		}
	};
}

# endif