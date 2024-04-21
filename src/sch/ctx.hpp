# pragma once

# include <cnt.hpp>
# include <opr.hpp>
# include <str.hpp>

# include <cstdint>
# include <mutex>
# include <iostream>
# include <deque>
# include <memory>
# include <thread>
# include <condition_variable>
# include <chrono>

# include <cuda.h>
# include <cudaTypedefs.h>
# include <c10/cuda/CUDAStream.h>

using namespace std;
using namespace c10::cuda;
using namespace chrono;

namespace FGPRS
{
	class Operation;
	class MyContainer;
	class MyStream;

	class MyContext
	{
	private:
		CUcontext _context;
		bool _default;
		mutable mutex* _pMutex;
		mutable mutex* _pQueueMutex;
		unique_lock<mutex> _lock;
		mutable condition_variable* cv;
		bool _changed = true;
		steady_clock::time_point _finishTime;
		vector<MyStream> _mainStreams, _secondaryStreams;
		shared_ptr<Operation> _headOperation;
		int _runningCount = 0;
		bool _stopDummies = false;
		thread _dummyThread;

	public:
		vector<shared_ptr<MyContainer>> highContainers, lowContainers, allContainers, dummyContainers;
		int smCount;
		static int mainStreamCount, secondaryStreamCount;
		int index = -1;
		vector<shared_ptr<Operation>>
			highLastDelayed, highLast, highDelayed, highOther,
			lowLastDelayed, lowLast, lowDelayed, lowOther;
		vector<shared_ptr<Operation>> running;
		double highUtilization, activeUtilization, overallUtilization;
		int missedCount = 0, acceptedCount = 0, skippedCount = 0;
		double acceptanceRate = 1;
		int remainingSecondaryStreams = 0;

		MyContext() {}
		MyContext(int index, int smCount, bool isDefault = false);
		bool initialize();
		bool select();
		static bool selectDefault();

		void selectHeadOperation();
		void queueOperation(shared_ptr<Operation> operation);
		void dequeueOperation(shared_ptr<Operation> operation);
		void releaseOperation(shared_ptr<Operation> operation);
		void finishOperation(shared_ptr<Operation> operation);
		steady_clock::time_point finishTime();

		void updateUtilization();
		void assignModule(shared_ptr<MyContainer> container);
		void removeModule(shared_ptr<MyContainer> container);
		void warmup();

		void runDummies(shared_ptr<MyContainer> operation);
		void stopDummies();
		void waitDummies();

		MyStream* getStream();
		MyStream* getSecondaryStream(MyStream* mainStream);
	};
}