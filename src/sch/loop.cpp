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
		Scheduler::selectDefaultContext();
		_container->forward(dummyInput);

		for (int j = 0; j < Scheduler::smOptions.size(); j++)
		{
			auto ctx = Scheduler::selectContextByIndex(j);
			ctx->select();
			_container->forward(dummyInput);
			ctx->release();
		}
	}

	Scheduler::selectDefaultContext();

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
		container->assignDeadline(period / 1000 * 0.8, level, 3, 0);

	else if (type == MPS_SCHEDULER || type == PMPS_SCHEDULER || type == PMPSO_SCHEDULER)
	{
		auto ctx = Scheduler::selectContextByIndex(index);
		ctx->select();
	}

	startTime = steady_clock::now();
	nextTime = startTime;

	while (!*stop)
	{
		if (type == PROPOSED_SCHEDULER)
			container->setAbsoluteDeadline(level, nextTime);

		nextTime += interval;

		frame++;
		// auto now = chrono::system_clock::now();
		// auto us = chrono::duration_cast<chrono::microseconds>(now.time_since_epoch()).count();
		// time_t now_c = chrono::system_clock::to_time_t(now);
		// char buffer[80];
		// strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", localtime(&now_c));
		// cout << buffer << "." << setfill('0') << setw(6) << (us % 1000000) << endl;

		// struct timespec ts;
		// clock_gettime(CLOCK_REALTIME, &ts);
		// time_t nowtime = ts.tv_sec;
		// struct tm* nowtm = localtime(&nowtime);
		// char tmbuf[64];
		// strftime(tmbuf, sizeof(tmbuf), "%Y-%m-%d %H:%M:%S", nowtm);
		// cout << "Current time: " << tmbuf << "." << ts.tv_nsec / 1000 << " microseconds" << endl;

		if (type == PROPOSED_SCHEDULER)
			container->schedule(name, *input, level);

		else
			container->forward(*input);

		if (steady_clock::now() > nextTime)
		{
			// cout << "          Time: " << ((duration_cast<microseconds>(steady_clock::now().time_since_epoch())).count() % 1000000) << endl;
			// cout << "Deadline missed: " << frame << "th " << name << endl;

			// if (steady_clock::now() > (nextTime + interval / 10))
			// 	cout << name << "->Delayed: " << duration_cast<microseconds>(steady_clock::now() - nextTime).count() << "us" << endl;

			// else
			// 	cout << name << "->Bevakhed: " << duration_cast<microseconds>(steady_clock::now() - nextTime).count() << "us" << endl;
			container->scheduleLogger->info("Delayed : {}us", duration_cast<microseconds>(steady_clock::now() - nextTime).count());

			// if ((nextTime + interval) < steady_clock::now())
			// 	printf("OHA!!!\n");

			continue;
		}

		// cout << "          Time: " << ((duration_cast<microseconds>(steady_clock::now().time_since_epoch())).count() % 1000000) << endl;
		// cout << "Reserved: " << frame << "th " << name << " " << duration_cast<microseconds>(nextTime - steady_clock::now()).count() << "us" << endl;
		container->scheduleLogger->info("Reserved: {}us", duration_cast<microseconds>(nextTime - steady_clock::now()).count());

		this_thread::sleep_until(nextTime);
	}
}

void Loop::start(Tensor* input, SchedulerType type, int level)
{
	_stop = false;

	_th = thread(run, _name, _container, input, _period, &_stop, level, _index, type);
}

void Loop::stop()
{
	_stop = true;
	_th.join();
}