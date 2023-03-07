# ifndef CTX_H
# define CTX_H

# include <operation.h>

# include <cstdint>
# include <mutex>
# include <iostream>
# include <deque>
# include <memory>

# include <cuda.h>
# include <cudaTypedefs.h>

using namespace std;

namespace FGPRS
{
	class MyContext
	{
	private:
		CUcontext _context;
		bool _default;
		mutable mutex* _pMutex;
		deque<Operation*> _queue;
		bool _changed = true;
		steady_clock::time_point _finishTime;


	public:
		bool busy = false;
		unsigned smCount;
		int index = -1;

		MyContext() {}
		MyContext(unsigned, int, bool = false);
		bool initialize();
		bool select();
		static bool selectDefault();
		bool release();
		void lock();
		void unlock();

		void queueOperation(Operation* operation);
		void dequeueOperation();
		steady_clock::time_point getFinishTime();
	};
}

# endif