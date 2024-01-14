# ifndef CTX_H
# define CTX_H

# include <opr.hpp>

# include <cstdint>
# include <mutex>
# include <iostream>
# include <deque>
# include <memory>

# include <cuda.h>
# include <cudaTypedefs.h>
# include <c10/cuda/CUDAStream.h>

using namespace std;
using namespace c10::cuda;

namespace FGPRS
{
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
		vector<CUDAStream> _streams;
		shared_ptr<Operation> _headOperation;

	public:
		vector<shared_ptr<MyContainer>> highContainers, lowContainers;
		static int streamCount;
		static int smCount;
		int index = -1;
		vector<shared_ptr<Operation>> highLast, highDelayed, highOther, lowLast, lowDelayed, lowOther;

		MyContext() {}
		MyContext(int index, bool isDefault = false);
		bool initialize();
		bool select();
		static bool selectDefault();
		bool release();

		void queueOperation(Operation* operation);
		void dequeueOperation(Operation* operation);
		steady_clock::time_point finishTime();
	};
}

# endif