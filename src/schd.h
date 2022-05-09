# ifndef SCHD_H
# define SCHD_H

# include <ctx.h>

# include <cstdint>

namespace FGPRS
{
	class Scheduler
	{
		public:
		const int MAX_CONTEXT_COUNT = 47;
		static int maxSmCount;

		public:
		static int _poolSize;
		static MyContext *_pContextPool;

		public:
		static bool initialize(int[], int);
		static bool initialize(int, int);
		static MyContext* selectContext(int);
		static bool selectDefaultContext();
		static bool releaseContext(MyContext);
		static bool destroyAll();
	};
}

# endif