# pragma once

# include <ctx.hpp>
# include <cnt.hpp>

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
		static int smCount, contextCount, maxSmCount, thresholdWindow;
		static MyContext* contextPool;
		static MyContext* _defaultContext;
		static vector<shared_ptr<ModuleTrackingRecord>> records;
		static vector<shared_ptr<MyContainer>> highContainers, lowContainers;
		static int missedCount, acceptedCount, skippedCount;
		static double acceptanceRate;

		static bool initialize(int contextCount, int smCount);
		static MyContext* selectContextByIndex(int index);
		static MyContext* selectDefaultContext();

		static float getTotalMemoryMB();
		static float getFreeMemoryMB();
		static float getTotalMemoryGB();
		static float getFreeMemoryGB();
		static float getMemoryPercentage();

		static void populateModulesByOrder(
			vector<shared_ptr<MyContainer>> highContainers,
			vector<shared_ptr<MyContainer>> lowContainers);
		static void populateModulesByUtilization(
			vector<shared_ptr<MyContainer>> highContainers,
			vector<shared_ptr<MyContainer>> lowContainers);
		static MyContext* findReplacementContext(shared_ptr<MyContainer> container, MyContext* previousContext);
		static void runDummies(shared_ptr<MyContainer> module);
		static void stopDummies();
		static void waitDummies();
	};
}