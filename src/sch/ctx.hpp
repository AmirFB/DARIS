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
		int lockCount = 0;
		mutable condition_variable* cv;
		bool _changed = true;
		steady_clock::time_point _finishTime;
		vector<c10::cuda::CUDAStream> _streams;

	public:
		bool busy = false;
		unsigned smCount;
		int index = -1;
		vector<Operation*> queue;// = vector<Operation*>(0);

		MyContext() {}
		MyContext(unsigned, int, bool = false);
		bool initialize();
		// c10::cuda::CUDAStream* select();
		// static c10::cuda::CUDAStream* selectDefault();
		bool select();
		static bool selectDefault();
		bool release();
		void lock();
		void unlock();
		bool isEmpty();

		void queueOperation(Operation* operation);
		void dequeueOperation(Operation* operation);
		steady_clock::time_point getFinishTime();
	};
}

# endif