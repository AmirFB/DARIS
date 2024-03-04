# pragma once

# include <str.hpp>
# include <opr.hpp>
# include <ctxd.hpp>
# include <trc.hpp>

# include <torch/torch.h>

# include <chrono>
# include <future>
# include <stdio.h>
# include <memory>

using namespace torch;
using namespace torch::nn;
using namespace at;

using namespace std;
using namespace chrono;

namespace FGPRS
{
	class Operation;
	class MyStream;

	class ParalleltialImpl : public Module
	{
	private:
		vector<shared_ptr<SequentialImpl>> _parallels;

	public:
		ParalleltialImpl(vector<shared_ptr<SequentialImpl>> parallels);
		Tensor forward(Tensor input);
	};
	TORCH_MODULE(Paralleltial);

	class MyContainer : public Module
	{
	private:
		time_point<steady_clock> _lastArrival;
		bool _stopDummy = false;
		thread _dummyThread;
		int _iterationCount = 0;

	public:
		bool highPriority = false;
		vector<shared_ptr<Operation>> operations;
		int interval, frequency;
		MyContext* currentContext;
		int operationCount;
		int inputSize;
		bool active = false;
		int missedCount = 0, acceptedCount = 0, skippedCount = 0;
		double acceptanceRate = 0;
		bool isDummy = false;

		ModuleTracker tracker;
		shared_ptr<ModuleTrackingRecord> currentRecord;
		string moduleName;
		int bcet, wcet, wret;
		double utilizationIsolated, utilizationPartitioned;

		MyContainer() {}
		MyContainer(const MyContainer& container);

		virtual void initialize(shared_ptr<MyContainer> container, string name, bool highPriority) {}
		void setFrequency(int frequency);
		void assignExecutionTime();
		void updateExecutionTime();
		void updateUtilization();
		void addOperation(shared_ptr<Operation> operation);
		void reset();
		virtual Tensor forward(Tensor input) { return input; }
		Tensor forwardRandom() { return forward(torch::rand({ 1, 3, inputSize, inputSize }).cuda()); }
		bool doesMeetDeadline();
		bool isFair();
		int admissionTest();
		void updateAcceptanceRate(bool accepted);
		virtual Tensor releaseOperations(Tensor input);
		virtual bool release(Tensor input);

		void runDummy();
		void stopDummy();
		void waitDummy();

		virtual void analyzeBCET(int warmup, int repeat);
		virtual void analyzeWCET(int warmup, int repeat);

		virtual void assignDeadline();
		void setAbsoluteDeadline(steady_clock::time_point start);

	private:
		virtual Tensor forwardDummy(Tensor input, MyStream* str);
		virtual void analyzeOperations(int warmup, int repeat, bool isWcet);
	};
}