# include <main.hpp>

# include <loop.hpp>

# include <cnt.hpp>
# include <schd.hpp>

# include <memory>
# include <chrono>
# include <thread>
# include <ctime>
# include <vector>
# include <random>

# include <c10/cuda/CUDACachingAllocator.h>
# include <cuda_runtime_api.h>
# include <nvToolsExt.h>

using namespace FGPRS;

using namespace torch;
using namespace torch::nn;

using namespace std;
using namespace chrono;

bool first = true;

Loop::Loop(shared_ptr<MyContainer> container)
	: _container(container), _name(container->moduleName),
	_frequency(container->frequency), _period(1000000 / container->frequency)
{
}

void run(shared_ptr<MyContainer> container, int timer)
{
	// cout << "Start " << container->moduleName << " thread" << endl;
	NoGradGuard no_grad;
	int frame = 0;
	steady_clock::time_point startTime, nextTime;
	auto interval = microseconds((int)round(container->interval));

	container->currentContext->select();
	Tensor input = torch::rand({ container->batchSize, 3, container->inputSize, container->inputSize }).cuda();

	srand(steady_clock::now().time_since_epoch().count());
	auto delay = rand() % (int)round(container->interval);
	startTime = steady_clock::now() + std::chrono::microseconds(delay);
	auto endTime = startTime + milliseconds(timer);

	nextTime = startTime + interval;

	std::this_thread::sleep_until(startTime);

	// cout << "Start " << container->moduleName << " with delay " << delay << "us" << endl;

	while (true)
	{
		frame++;
		// cout << container->moduleName << " frame " << frame << " start." << endl;
		container->setAbsoluteDeadline(nextTime);

		auto output = container->release(input);
		nextTime += interval;

		if (steady_clock::now() > nextTime)
		{
			// cout << (container->highPriority ? "H" : "L") << "Container "
			// 	<< container->moduleName << " missed deadline (frame " << frame << ")." << endl;
			// container->missedCount++;
			container->currentRecord->missed = true;
		}

		// cout << container->moduleName << " frame " << frame << " end." << endl;

		if (steady_clock::now() > endTime)
			break;

		this_thread::sleep_until(nextTime);
	}

	cout << "End " << container->moduleName << endl;
}

void Loop::start(int timer)
{
	_th = thread(run, _container, timer);
}

void Loop::wait()
{
	_th.join();
}