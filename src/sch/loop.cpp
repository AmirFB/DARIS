# include <loop.hpp>

# include <cnt.hpp>
# include <schd.hpp>

# include <memory>
# include <chrono>
# include <thread>
# include <ctime>
# include <sys/time.h>
# include <pthread.h>

# include <c10/cuda/CUDACachingAllocator.h>
# include <cuda_runtime_api.h>
# include <nvToolsExt.h>

using namespace FGPRS;

using namespace torch;
using namespace torch::nn;

using namespace std;
using namespace chrono;

bool first = true;

Loop::Loop(string name, shared_ptr<MyContainer> container, double frequency, int index)
	: _name(name), _container(container), _frequency(frequency), _period(1000000000 / frequency), _index(index)
{
}

void Loop::initialize(int deadlineContextIndex, Tensor dummyInput, SchedulerType type, int level)
{
	MyContext::selectDefault();
	_container->eval();
	_container->to(kCUDA);

	_container->initLoggers(_name);

	// auto stream = at::cuda::getStreamFromPool(false, 0);

	for (int j = Scheduler::smOptions.size() - 1; j >= 0; j--)
	{
		auto ctx = Scheduler::selectContextByIndex(j);
		ctx->select();

		if (type == PROPOSED_SCHEDULER && (j != (Scheduler::smOptions.size() - 1) && Scheduler::noDefault == true))
		{
			auto stream = at::cuda::getStreamFromPool(false, ctx->index);
			at::cuda::setCurrentCUDAStream(stream);

			for (int i = 0; i < 10; i++)
			{
				_container->forward(dummyInput);

				if (j != (Scheduler::smOptions.size() - 1))
					stream.synchronize();
			}
		}

		else
			for (int i = 0; i < 10; i++)
				_container->forward(dummyInput);

		cuCtxSynchronize();
		ctx->release();
	}

	_container->assignOperations();

	if (type == PROPOSED_SCHEDULER)
	{
		auto id = nvtxRangeStartA(_name.c_str());
		_container->analyze(10, 25, dummyInput, level);
		nvtxRangeEnd(id);

		if (first)
		{
			// _container->clearAnalyzeLogger(_name);
			id = nvtxRangeStartA(_name.c_str());
			_container->analyze(1, 1, dummyInput, level);
			nvtxRangeEnd(id);
			first = false;
		}

		_container->assignExecutionTime(level, deadlineContextIndex, 0);
		_container->assignDeadline(_period / 1000 * 0.95, level, deadlineContextIndex, 0);
	}
}

void run(
	string name, shared_ptr<MyContainer> container, Tensor* input,
	double period, bool* stop, int level, int index,
	SchedulerType type, bool logIt)
{
	NoGradGuard no_grad;
	int frame = 0;
	steady_clock::time_point startTime, nextTime;
	auto interval = nanoseconds((int)round(period));
	// container->clearScheduleLogger(name);

	if (type == MPS_SCHEDULER || type == PMPS_SCHEDULER || type == PMPSO_SCHEDULER)
	{
		auto ctx = Scheduler::selectContextByIndex(index);
		ctx->select();
	}

	// c10::cuda::CUDAStream stream = at::cuda::getStreamFromPool(false, index);
	// at::cuda::setCurrentCUDAStream(stream);

	// cudaProfilerSetStringName(threadId, name.c_str());
	// pthread_t nativeHandle = myThread.native_handle();
	// pthread_setname_np(pthread_self(), name.c_str());

	startTime = steady_clock::now();
	// nextTime = startTime + nanoseconds((int)round(period * 0.25));
	nextTime = startTime + milliseconds(2);

	container->meets = 0;
	container->missed = 0;

	steady_clock::time_point dummyNow;

	while (!*stop)
	{
		auto id = nvtxRangeStartA((name + " " + to_string(frame)).c_str());

		if (type == PROPOSED_SCHEDULER)
			container->setAbsoluteDeadline(level, nextTime, 0);

		nextTime += interval;
		frame++;

		// cout << name << " " << frame << endl;
		if (type == PROPOSED_SCHEDULER)
			container->schedule(*input, level);

		else
		{
			container->forward(*input);
			cuCtxSynchronize();
		}
		// cout << name << " " << frame << endl;
		dummyNow = steady_clock::now();

		if (dummyNow > nextTime)
		{
			if (logIt)
				container->scheduleLogger->info("Delayed : {}us", duration_cast<microseconds>(dummyNow - nextTime).count());

			while (dummyNow > (nextTime + interval))
			{
				nextTime += interval;
			}

			container->missed++;
			container->delayed = true;
		}

		else
		{
			if (logIt)
				container->scheduleLogger->info("Reserved: {}us", duration_cast<microseconds>(nextTime - dummyNow).count());
			container->meets++;
			container->delayed = false;
			// c10::cuda::CUDACachingAllocator::emptyCache();
		}

		nvtxRangeEnd(id);

		this_thread::sleep_until(nextTime);
		// cout << name << " " << frame << "\tstop: " << *stop << endl;
	}

	string temp = name +
		"\n\tCompleted: " + to_string((container->meets + container->missed) * 1 / (ceil(1000000000.0 / period / 1)) * 100) +
		"%" + "\tMissed   : " + to_string((1 - container->meets * 1 / (ceil(1000000000.0 / period / 1))) * 100) + "%\n";
	cout << temp;
}

void Loop::start(Tensor* input, SchedulerType type, int level, bool logIt)
{
	_stop = false;

	_th = thread(run, _name, _container, input, _period, &_stop, level, _index, type, logIt);
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