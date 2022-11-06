# ifndef SCHD_H
# define SCHD_H

# include <ctx.h>
# include <chrono>

# include <cstdint>

namespace FGPRS
{
	class Scheduler
	{
		public:
		const int MAX_CONTEXT_COUNT = 48;
		static int maxSmCount;
		static int *smOptions;
		static int poolSize;

		private:
		static MyContext *_pContextPool;
		static MyContext *_defaultContext;


		public:
		static bool initialize(int[], int);
		static bool initialize(int, int, int);
		static MyContext* selectContext(int);
		static MyContext* selectContextByIndex(int index);
		static MyContext* selectContextByQueueLoad(double* executionTime);
		static bool selectDefaultContext();
		static bool releaseContext(MyContext);
		static bool destroyAll();

		static float getTotalMemoryMB();
		static float getFreeMemoryMB();
		static float getTotalMemoryGB();
		static float getFreeMemoryGB();
		static float getMemoryPercentage();
	};
}

# endif