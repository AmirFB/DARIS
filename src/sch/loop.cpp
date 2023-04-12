# include <loop.hpp>

# include <cnt.hpp>
# include <schd.hpp>
# include <log.hpp>

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

void Loop::initialize(int deadlineContextIndex, Tensor dummyInput, int level)
{
	_container->eval();
	_container->to(kCUDA);

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

	_container->analyze(1, 3, dummyInput, level);
	// couts << endl << endl;
	// _container->analyze(5, 10, dummyInput, 2);
	// couts << endl << endl;
	// _container->analyze(5, 10, dummyInput, 1);
	cout << "Shall we?\n";
	_container->assignExecutionTime(level, deadlineContextIndex, 0);
}

void run(
	string name, shared_ptr<MyContainer> container, Tensor* input,
	double period, bool* stop, int level, int index,
	SchedulerType type)
{
	int frame = 0;
	steady_clock::time_point startTime, nextTime;
	auto interval = nanoseconds((int)round(period));

	if (type == PROPOSED_SCHEDULER)
	{
		container->assignDeadline(period / 1000 * 0.9, level, 3, 0);
		// container->assignDeadline(period / 1000 * 0.9, 2, 3, 0);
		// container->assignDeadline(period / 1000 * 0.9, 1, 3, 0);
	}

	else if (type == MPS_SCHEDULER || type == PMPS_SCHEDULER || type == PMPS_SCHEDULER)
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
		// couts << "          Next: " << ((duration_cast<microseconds>(nextTime.time_since_epoch())).count() % 1000000) << endl;
		// couts << "          Time: " << ((duration_cast<microseconds>(startTime.time_since_epoch())).count() % 1000000) << endl;
		// couts << "          Time: " << ((duration_cast<milliseconds>(steady_clock::now().time_since_epoch())).count() % 1000000) << endl;

		// auto now = std::chrono::system_clock::now();
		// auto us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
		// std::time_t now_c = std::chrono::system_clock::to_time_t(now);
		// char buffer[80];
		// std::strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", std::localtime(&now_c));
		// std::couts << buffer << "." << std::setfill('0') << std::setw(6) << (us % 1000000) << std::endl;

		// struct timespec ts;
		// clock_gettime(CLOCK_REALTIME, &ts);
		// time_t nowtime = ts.tv_sec;
		// struct tm* nowtm = localtime(&nowtime);
		// char tmbuf[64];
		// strftime(tmbuf, sizeof(tmbuf), "%Y-%m-%d %H:%M:%S", nowtm);
		// std::couts << "Current time: " << tmbuf << "." << ts.tv_nsec / 1000 << " microseconds" << std::endl;

		if (type == PROPOSED_SCHEDULER)
			container->schedule(name, *input, level);

		else
			container->forward(*input);

		if (steady_clock::now() > nextTime)
		{
			// couts << "          Time: " << ((duration_cast<microseconds>(steady_clock::now().time_since_epoch())).count() % 1000000) << endl;
			// couts << "Deadline missed: " << frame << "th " << name << endl;

			// if (steady_clock::now() > (nextTime + interval / 10))
			// 	couts << name << "->Delayed: " << duration_cast<microseconds>(steady_clock::now() - nextTime).count() << "us" << endl;

			// else
			// 	couts << name << "->Bevakhed: " << duration_cast<microseconds>(steady_clock::now() - nextTime).count() << "us" << endl;

			if ((nextTime + interval) < steady_clock::now())
			{
				printfs("OHA!!!\n");
				break;
			}

			continue;
		}

		// couts << "          Time: " << ((duration_cast<microseconds>(steady_clock::now().time_since_epoch())).count() % 1000000) << endl;
		// couts << "Reserved: " << frame << "th " << name << " " << duration_cast<microseconds>(nextTime - steady_clock::now()).count() << "us" << endl;

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