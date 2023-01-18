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

		MyContext();
		MyContext(unsigned);
		bool initialize();
		bool select(double duration);
		static bool selectDefault();
		bool release(double duration);
		bool destroy();
		void lock();
		void unlock();
	};
}

# endif