# include <loop.hpp>

# include <cnt.hpp>
# include <schd.hpp>

# include <memory>
# include <chrono>
# include <thread>

using namespace FGPRS;

using namespace torch;
using namespace torch::nn;

using namespace std;
using namespace chrono;

Loop::Loop(string name, shared_ptr<MyContainer> container, double frequency, int index)
	: _name(name), _container(container), _frequency(frequency), _period(1000000000 / frequency), _index(index)
{
}

void Loop::initialize(int deadlineContextIndex, Tensor dummyInput)
{
	_container->eval();
	_container->to(kCUDA);

	for (int i = 0; i < 10; i++)
	{
		Scheduler::selectDefaultContext();
		_container->forward(dummyInput);

		for (int j = 0; j < 4; j++)
		{
			auto ctx = Scheduler::selectContextByIndex(j);
			ctx->select();
			_container->forward(dummyInput);
			ctx->release();
		}
	}

	Scheduler::selectDefaultContext();

# if SCHEDULER_TYPE == PROPOSED_SCHEDULER

	_container->assignOperations();

	_container->analyze(1, 1, dummyInput, 3);
	cout << endl << endl;
	_container->analyze(1, 1, dummyInput, 2);
	cout << endl << endl;
	_container->analyze(1, 1, dummyInput, 1);

	_container->assignExecutionTime(deadlineContextIndex);
# endif
}

void run(string name, shared_ptr<MyContainer> container, Tensor* input, double period, bool* stop, int level, int index)
{
	int frame = 0;
	steady_clock::time_point startTime, nextTime;
	auto interval = nanoseconds((int)round(period));

# if SCHEDULER_TYPE == PROPOSED_SCHEDULER
	container->assignDeadline(period / 1000, 3, 3, 0);
	container->assignDeadline(period / 1000, 2, 3, 0);
	container->assignDeadline(period / 1000, 1, 3, 0);
# endif

	startTime = steady_clock::now();
	nextTime = startTime;

	while (!*stop)
	{
		frame++;
		cout << "          Next: " << ((duration_cast<microseconds>(nextTime.time_since_epoch())).count() % 1000000) << endl;
		cout << "          Time: " << ((duration_cast<microseconds>(steady_clock::now().time_since_epoch())).count() % 1000000) << endl;
		cout << "Scheduling Started     " << name << " " << frame << "th" << endl;

# if SCHEDULER_TYPE == PROPOSED_SCHEDULER
		container->setAbsoluteDeadline(level, nextTime);
		container->schedule(name, *input, level);
# endif

# if SCHEDULER_TYPE != PROPOSED_SCHEDULER
		container->forward(*input);
# endif

		nextTime += interval;

		if (steady_clock::now() > nextTime)
		{
			cout << "          Time: " << ((duration_cast<microseconds>(steady_clock::now().time_since_epoch())).count() % 1000000) << endl;
			cout << "Deadline missed: " << frame << "th " << name << endl;
			cout << "Delayed: " << duration_cast<microseconds>(steady_clock::now() - nextTime).count() << "us" << endl;

			while ((nextTime + interval) < steady_clock::now())
			{
				cout << "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_\n";
				nextTime += interval;
			}

			continue;
		}

		cout << "          Time: " << ((duration_cast<microseconds>(steady_clock::now().time_since_epoch())).count() % 1000000) << endl;
		cout << "Reserved: " << frame << "th " << name << " " << duration_cast<microseconds>(nextTime - steady_clock::now()).count() << "us" << endl;

		this_thread::sleep_until(nextTime);
	}
}

void Loop::start(Tensor* input, int level)
{
	_stop = false;

	_th = thread(run, _name, _container, input, _period, &_stop, level, _index);
}

void Loop::stop()
{
	_stop = true;
	_th.join();
}