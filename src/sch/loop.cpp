# include <loop.h>

# include <cnt.h>
# include <schd.h>

# include <memory>
# include <chrono>
# include <thread>

using namespace FGPRS;

using namespace torch;
using namespace torch::nn;

using namespace std;
using namespace chrono;

Loop::Loop(shared_ptr<MyContainer> container, double frequency)
	: _container(container), _frequency(frequency), _period(1000000000 / frequency)
{
}
// Loop::Loop(shared_ptr<MyContainer> container, double period)
// 	: _container(container), _period(period * 1000000), _frequency(1000 / period)
// {
// }

void Loop::initialize(int deadlineContextIndex, Tensor dummyInput)
{
	_container->eval();
	_container->to(kCUDA);
	_container->assignOperations();

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

	_container->analyze(5, 10, dummyInput, 3);
	cout << endl << endl;
	_container->analyze(5, 10, dummyInput, 2);
	cout << endl << endl;
	_container->analyze(5, 10, dummyInput, 1);

	_container->assignExecutionTime(deadlineContextIndex);
}

void run(shared_ptr<MyContainer> container, Tensor* input, double period, bool* stop, int level)
{
	int frame = 0;
	steady_clock::time_point startTime, nextTime;
	auto interval = nanoseconds((int)round(period));

	container->assignDeadline(period / 1000, 3, 3, 0);
	container->assignDeadline(period / 1000, 2, 3, 0);
	container->assignDeadline(period / 1000, 1, 3, 0);

	startTime = steady_clock::now();
	nextTime = startTime;

	while (!*stop)
	{
		frame++;
		container->setAbsoluteDeadline(level, nextTime);
		container->schedule(*input, level);

		nextTime += interval;

		if (steady_clock::now() > nextTime)
		{
			cout << "Deadline missed: " << container->name() << " (" << frame << ")\n";
			cout << "Now : " << duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count() << endl;
			cout << "Next: " << duration_cast<nanoseconds>(nextTime.time_since_epoch()).count() << endl << endl;
			continue;
		}

		this_thread::sleep_until(nextTime);
		cout << frame << "th frame is done.\n";
	}
}

void Loop::start(Tensor* input, int level)
{
	_stop = false;
	_th = thread(run, _container, input, _period, &_stop, level);
}

void Loop::stop()
{
	_stop = true;
	_th.join();
}