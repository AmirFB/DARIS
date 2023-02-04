# ifndef SCHD_H
# define SCHD_H

# include <ctx.h>

# include <chrono>
# include <cstdint>
# include <vector>

# include <torch/torch.h>

using namespace torch;
using namespace torch::nn;

namespace FGPRS
{
	class Scheduler
	{
		public:
		const int MAX_CONTEXT_COUNT = 48;
		static int maxSmCount;
		static vector<int> smOptions;
		static bool _stopDummy;

	private:
		static vector<MyContext> _contextPool;
		static MyContext _defaultContext;
		static Sequential _dummyModule;
		static Tensor _dummyInput;
		static Sequential _dummyModule2;
		static Tensor _dummyInput2;

		public:
		static bool initialize(int[], int);
		static MyContext selectContext(int);
		static MyContext selectContextByIndex(int index);
		static MyContext selectContextByQueueLoad(double* executionTime);
		static bool selectDefaultContext();
		static bool releaseContext(MyContext);
		static bool destroyAll();

		static vector<MyContext> getAllContexts();

		static float getTotalMemoryMB();
		static float getFreeMemoryMB();
		static float getTotalMemoryGB();
		static float getFreeMemoryGB();
		static float getMemoryPercentage();

		static void dummyFunction(MyContext ctx);
		static int startDummy(int);
		static void stopDummy();
	};
}

# endif