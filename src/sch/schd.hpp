# ifndef SCHD_H
# define SCHD_H

# include <ctx.hpp>

# include <chrono>
# include <cstdint>
# include <vector>
# include <future>

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
		static MyContext* _contextPool;
		static MyContext _defaultContext;
		static Sequential* _dummyModule;
		static Tensor* _dummyInput;
		static future<void>* _th;

	public:
		static bool initialize(int[], int);
		static MyContext* selectContext(int);
		static MyContext* selectContextByIndex(int index);
		static MyContext* selectDefaultContext();
		static bool releaseContext(MyContext);

		static vector<MyContext> getAllContexts();

		static float getTotalMemoryMB();
		static float getFreeMemoryMB();
		static float getTotalMemoryGB();
		static float getFreeMemoryGB();
		static float getMemoryPercentage();

		static void dummyFunction(MyContext* ctx, Sequential* mod, Tensor* in);
		static void startDummy(MyContext* ctx);
		static void stopDummy();

		static MyContext* getMinimalContext(Operation* operation);
		static MyContext* getFastestContext(Operation* operation);
	};
}

# endif