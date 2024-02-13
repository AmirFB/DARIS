# include <main.hpp>
# include <schd.hpp>

# include <iostream>
# include <thread>
# include <future>
# include <ranges>
# include <vector>
# include <unistd.h>
# include <mutex>

# include <cuda.h>
# include <cudaTypedefs.h>
# include <cuda_runtime.h>
# include <cuda_runtime_api.h>
# include <nvToolsExt.h>

# include <torch/torch.h>

using namespace FGPRS;

using namespace std;

using namespace torch;
using namespace torch::nn;

MyContext* Scheduler::contextPool;
MyContext* Scheduler::_defaultContext;

int Scheduler::smCount;
int Scheduler::contextCount = 0;
int Scheduler::maxSmCount;
int Scheduler::thresholdWindow;

vector<shared_ptr<MyContainer>> Scheduler::highContainers, Scheduler::lowContainers;

int Scheduler::missedCount = 0, Scheduler::acceptedCount = 0;
double Scheduler::acceptanceRate = 1;

bool Scheduler::initialize(int contextCount, int smCount)
{
	bool result = true;
	cudaDeviceProp prop;

	cudaGetDeviceProperties(&prop, 0);
	maxSmCount = prop.multiProcessorCount;

	Scheduler::contextCount = contextCount;
	contextPool = new MyContext[contextCount + 1];

	contextPool[contextCount] = MyContext(contextCount, maxSmCount, true);
	_defaultContext = &contextPool[contextCount];
	result &= contextPool[contextCount].initialize();

	for (int i = 0; i < contextCount; i++)
	{
		contextPool[i] = MyContext(i, smCount, false);
		result &= contextPool[i].initialize();
	}

	return result;
}

MyContext* Scheduler::selectContextByIndex(int index)
{
	return &contextPool[index];
}

MyContext* Scheduler::selectDefaultContext()
{
	// cout << "Context index: " << contextCount << endl;
	return &contextPool[contextCount];
}

float Scheduler::getTotalMemoryMB()
{
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	return total / 1024. / 1024.;
}

float Scheduler::getTotalMemoryGB()
{
	return Scheduler::getTotalMemoryMB() / 1024;
}

float Scheduler::getFreeMemoryMB()
{
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	return free / 1024. / 1024.;
}

float Scheduler::getFreeMemoryGB()
{
	return Scheduler::getFreeMemoryMB() / 1024;
}

float Scheduler::getMemoryPercentage()
{
	return Scheduler::getFreeMemoryMB() / Scheduler::getTotalMemoryMB() * 100;
}

mutex globalMutex;

void Scheduler::populateModules(
	vector<shared_ptr<MyContainer>> highContainers,
	vector<shared_ptr<MyContainer>> lowContainers)
{
	int index = 0, firstLow;
	bool first = true;

	for (auto high : highContainers)
	{
		Scheduler::highContainers.push_back(high);
		contextPool[index].select();
		contextPool[index].assignModule(high);
		high->currentContext = &contextPool[index];

		high->to(kCUDA);
		high->eval();
		high->forwardRandom();

		index = (index + 1) % contextCount;
	}

	firstLow = index;

	for (auto low : lowContainers)
	{
		Scheduler::lowContainers.push_back(low);
		contextPool[index].select();
		contextPool[index].assignModule(low);
		low->currentContext = &contextPool[index];

		low->to(kCUDA);
		low->eval();
		low->forwardRandom();

		index = (index + 1) % contextCount;

		if (first & index == 0)
		{
			first = false;
			index = firstLow;
		}
	}

	for (int i = 0; i < contextCount; i++)
	{
		cout << "Context " << i << " has "
			<< contextPool[i].highContainers.size() << " high priority modules and "
			<< contextPool[i].lowContainers.size() << " low priority modules." << endl;

		contextPool[i].warmup();
	}
}

void Scheduler::runDummies(shared_ptr<MyContainer> module)
{
	for (int i = 0; i < contextCount; i++)
		contextPool[i].runDummies(module);
}

void Scheduler::stopDummies()
{
	for (int i = 0; i < contextCount; i++)
		contextPool[i].stopDummies();
}

void Scheduler::waitDummies()
{
	for (int i = 0; i < contextCount; i++)
		contextPool[i].waitDummies();
}