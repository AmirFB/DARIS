# ifndef CTX_H
# define CTX_H

# include <cstdint>

# include <cuda.h>
# include <cudaTypedefs.h>

namespace FGPRS
{
	class MyContext
	{
		public:
		CUcontext _context;
		bool _default;

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
	};
}

# endif