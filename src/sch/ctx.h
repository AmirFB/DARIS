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
		deque<shared_ptr<Operation>> _queue;
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
		bool destroy();
		void lock();
		void unlock();

		void addOperation(shared_ptr<Operation> operation);
		void removeOperation();
		steady_clock::time_point getFinishTime();
	};

	// struct ContextData
	// {
	// 	MyContext* context;
	// 	double isolatedExecutionTime, occupiedExecutionTime;
	// 	int smCount;

	// 	ContextData(MyContext* context)
	// 	{
	// 		this->context = context;
	// 		smCount = context->smCount;
	// 		isolatedExecutionTime = 0;
	// 		occupiedExecutionTime = 0;
	// 	}

	// 	ContextData(MyContext* context, double isolatedExecutionTime, double occupiedExecutionTime)
	// 	{
	// 		this->context = context;
	// 		smCount = context->smCount;
	// 		this->isolatedExecutionTime = isolatedExecutionTime;
	// 		this->occupiedExecutionTime = occupiedExecutionTime;
	// 	}

	// 	void stackExecutionTime(ContextData ctxData)
	// 	{
	// 		this->isolatedExecutionTime += ctxData.isolatedExecutionTime;
	// 		this->occupiedExecutionTime += ctxData.occupiedExecutionTime;
	// 	}
	// };
}

# endif