# include <main.hpp>

# include <loop.hpp>

# include <cnt.hpp>
# include <schd.hpp>

# include <memory>
# include <chrono>
# include <thread>
# include <ctime>
# include <sys/time.h>
# include <pthread.h>
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

std::mutex zadMutex;

void run(shared_ptr<MyContainer> container, int timer)
{
	cout << "Start " << container->moduleName << " thread" << endl;
	NoGradGuard no_grad;
	int frame = 0;
	steady_clock::time_point startTime, nextTime;
	auto interval = microseconds((int)round(container->interval));

	container->currentContext->select();
	Tensor input = torch::rand({ 1, 3, container->inputSize, container->inputSize }).cuda();

	srand(steady_clock::now().time_since_epoch().count());
	auto delay = rand() % (int)round(container->interval);
	startTime = steady_clock::now() + std::chrono::microseconds(delay);
	auto endTime = startTime + milliseconds(timer);

	nextTime = startTime + interval;

	std::this_thread::sleep_until(startTime);

	cout << "Start " << container->moduleName << " with delay " << delay << "us" << endl;

	while (true)
	{
		container->setAbsoluteDeadline(nextTime);

		auto output = container->release(input);
		nextTime += interval;

		if (output == false)
		{
			// cout << "Container " << container->moduleName << " skipped." << endl;
		}

		else
		{
			frame++;

			// {
			// 	lock_guard<mutex> lock(zadMutex);
			// cout << "Container " << container->moduleName << (container->highPriority ? " H" : " L") << "frame " << frame << " finished" << endl;
			// }

			if (steady_clock::now() > nextTime)
			{
				cout << (container->highPriority ? "H" : "L") << "Container " << container->moduleName << " missed deadline" << endl;
				container->missedCount++;
				container->currentRecord->missed = true;

				// while (steady_clock::now() > (nextTime + interval * 0.1))
				// {
				// 	nextTime += interval;
				// }
			}
		}

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