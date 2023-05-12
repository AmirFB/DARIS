# include <loop.hpp>

# include <cnt.hpp>
# include <schd.hpp>

# include <memory>
# include <chrono>
# include <thread>
# include <ctime>
# include <sys/time.h>

using namespace FGPRS;

using namespace torch;
using namespace torch::nn;

using namespace std;
using namespace chrono;

Loop::Loop(string name, shared_ptr<MyContainer> container, double frequency, int index)
	: _name(name), _container(container), _frequency(frequency), _period(1000000000 / frequency), _index(index)
{
}

void Loop::initialize(int deadlineContextIndex, Tensor dummyInput, SchedulerType type, int level)
{
	_container->eval();
	_container->to(kCUDA);

	_container->initLoggers(_name);

	for (int i = 0; i < 10; i++)
	{
		for (int j = Scheduler::smOptions.size() - 1; j >= 0; j--)
		{
			auto ctx = Scheduler::selectContextByIndex(j);
			ctx->select();

			if (type == PROPOSED_SCHEDULER)
			{
				auto stream = at::cuda::getStreamFromPool(false, ctx->index);
				at::cuda::setCurrentCUDAStream(stream);
			}

			_container->forward(dummyInput);
			ctx->release();
		}
	}

	_container->assignOperations();

	if (type == PROPOSED_SCHEDULER)
	{
		_container->analyze(5, 10, dummyInput, level);
		_container->assignExecutionTime(level, deadlineContextIndex, 0);
	}
}

void run(
	string name, shared_ptr<MyContainer> container, Tensor* input,
	double period, bool* stop, int level, int index,
	SchedulerType type)
{
	NoGradGuard no_grad;
	int frame = 0;
	steady_clock::time_point startTime, nextTime;
	auto interval = nanoseconds((int)round(period));
	container->clearScheduleLogger(name);

	if (type == PROPOSED_SCHEDULER)
		container->assignDeadline(period / 1000 * 0.80, level, 3, 0);

	else if (type == MPS_SCHEDULER || type == PMPS_SCHEDULER || type == PMPSO_SCHEDULER)
	{
		auto ctx = Scheduler::selectContextByIndex(index);
		ctx->select();
	}

	startTime = steady_clock::now();
	nextTime = startTime;

	container->meets = 0;
	container->missed = 0;

	while (!*stop)
	{
		if (type == PROPOSED_SCHEDULER)
			container->setAbsoluteDeadline(level, nextTime);

		nextTime += interval;
		frame++;

		if (type == PROPOSED_SCHEDULER)
			container->schedule(*input, level);

		else
			container->forward(*input);

		if (steady_clock::now() > nextTime)
		{
			container->scheduleLogger->info("Delayed : {}us", duration_cast<microseconds>(steady_clock::now() - nextTime).count());

			while (steady_clock::now() > (nextTime + interval))
			{
				nextTime += interval;
			}

			container->missed++;
		}

		else
		{
			container->scheduleLogger->info("Reserved: {}us", duration_cast<microseconds>(nextTime - steady_clock::now()).count());
			container->meets++;
		}

		this_thread::sleep_until(nextTime);
	}

	string temp = name + "\n\tCompleted: " + to_string((container->meets + container->missed) * period / 1000000000.0 * 100) + "%"
		+ "\tMissed   : " + to_string((1 - container->meets * period / 1000000000.0) * 100) + "%";
	cout << temp << endl;
	// cout << name << endl
	// 	<< "\tCompleted: " << ((container->meets + container->missed) * period / 1000000000 * 100) << "%"
	// 	<< "\tMissed   : " << (1 - container->meets * period / 1000000000) * 100 << "%" << endl;
}

void Loop::start(Tensor* input, SchedulerType type, int level)
{
	_stop = false;

	_th = thread(run, _name, _container, input, _period, &_stop, level, _index, type);
}

void Loop::stop()
{
	_stop = true;
}

void Loop::wait()
{
	_stop = true;
	_th.join();
}