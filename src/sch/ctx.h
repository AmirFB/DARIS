# ifndef CTX_H
# define CTX_H

# include <cstdint>
# include <mutex>
# include <iostream>

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

	public:
		bool busy = false;
		unsigned smCount;
		unsigned long queueDuration;
		int index = -1;

		MyContext() {}
		MyContext(unsigned, int, bool = false);
		bool initialize();
		bool select();
		static bool selectDefault();
		bool release();
		bool destroy();
		void lock();
		void unlock();
	};

	struct ContextData
	{
		MyContext* context;
		double isolatedExecutionTime, occupiedExecutionTime;
		int smCount;

		ContextData(MyContext* context)
		{
			this->context = context;
			smCount = context->smCount;
			isolatedExecutionTime = 0;
			occupiedExecutionTime = 0;
		}

		ContextData(MyContext* context, double isolatedExecutionTime, double occupiedExecutionTime)
		{
			this->context = context;
			smCount = context->smCount;
			this->isolatedExecutionTime = isolatedExecutionTime;
			this->occupiedExecutionTime = occupiedExecutionTime;
		}

		void stackExecutionTime(ContextData ctxData)
		{
			this->isolatedExecutionTime += ctxData.isolatedExecutionTime;
			this->occupiedExecutionTime += ctxData.occupiedExecutionTime;
		}
	};
}

# endif