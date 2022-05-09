# ifndef CTX_H
# define CTX_H

# include <cstdint>

# include <cuda.h>
# include <cudaTypedefs.h>

namespace FGPRS
{
	class MyContext
	{
		private:
		CUexecAffinityParam_v1 _affinity;
		CUcontext _context;

		public:
		bool busy = false;
		unsigned smCount;

		MyContext(unsigned);
		bool initialize();
		bool select();
		static bool selectDefault();
		bool release();
		bool destroy();
	};
}

# endif