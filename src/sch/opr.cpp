# include <opr.hpp>

# include <ctxd.hpp>
# include <schd.hpp>
# include <log.hpp>

# include <torch/torch.h>

# include <chrono>
# include <iostream>
# include <unistd.h>
# include <future>
# include <cmath>

using namespace std;
using namespace std::chrono;
using namespace FGPRS;

double Operation::exceptionThreshold = 0.15;

string Operation::getName() { return _name; }
string Operation::getFullName() { return _fullName; }

void Operation::setName(string name)
{
	_name = name;
	_fullName = name;
}

void Operation::setParentName(string parentName)
{
	if (_lastParentName != parentName)
	{
		_fullName = parentName + "->" + _fullName;
		_lastParentName = parentName;
	}
}

Tensor Operation::analyze(int warmup, int repeat, Tensor input)
{
	Tensor output;
	bool first = true;
	steady_clock::time_point t1, t2, tStart, tEnd, tNow;
	int countIsolated, countOccupied;

	predictability = 0;
	isolatedScalability = 0;
	occupiedScalability = 0;

	cout << _fullName << ":" << endl;

	Scheduler::selectDefaultContext();
	contextData.clear();

	tStart = steady_clock::now();
	tEnd = tStart + milliseconds(warmup);

	for (int i = 0; i < warmup; i++)
		output = sequential->forward(input);

	for (auto sm : Scheduler::smOptions)
	{
		auto ctx = Scheduler::selectContext(sm);
		ctx->select();

		tStart = steady_clock::now();

		for (int i = 0; i < repeat; i++)
			output = sequential->forward(input);

		tNow = steady_clock::now();

		duration<double> d1 = tNow - tStart;

		ctx->release();
		Scheduler::startDummy(ctx);
		usleep(1000);
		ctx->select();

		countOccupied = 0;
		tStart = steady_clock::now();
		tEnd = tStart + milliseconds(repeat);

		tStart = steady_clock::now();

		for (int i = 0; i < repeat; i++)
			output = sequential->forward(input);

		tNow = steady_clock::now();

		duration<double> d2 = tNow - tStart;

		Scheduler::stopDummy();
		ctx->release();

		contextData.push_back(ContextData(ctx, d1.count() / repeat * 1000000, d2.count() / repeat * 1000000));
		cout << "\t" << ctx->smCount << "\t" << contextData.back().isolatedExecutionTime << "us"
			<< ", " << contextData.back().occupiedExecutionTime << "us";

		if (!first)
		{
			contextData.back().isolatedExecutionTime =
				min(contextData.back().isolatedExecutionTime, contextData[contextData.size() - 2].isolatedExecutionTime);
			contextData.back().occupiedExecutionTime =
				min(contextData.back().occupiedExecutionTime, contextData[contextData.size() - 2].occupiedExecutionTime);
		}

		predictability += 1 - (contextData.back().occupiedExecutionTime - contextData.back().isolatedExecutionTime) / contextData.back().occupiedExecutionTime;

		if (first)
		{
			first = false;
			continue;
		}

		double desired, isolatedGain, occupiedGain;

		desired = (double)contextData.back().smCount / contextData.end()[-2].smCount;
		isolatedGain = contextData.end()[-2].isolatedExecutionTime / contextData.back().isolatedExecutionTime;
		occupiedGain = contextData.end()[-2].occupiedExecutionTime / contextData.back().occupiedExecutionTime;

		isolatedScalability += max((isolatedGain - 1) / (desired - 1), 0.0);
		occupiedScalability += max((occupiedGain - 1) / (desired - 1), 0.0);
	}

	predictability /= 4;
	isolatedScalability /= 3;
	occupiedScalability /= 3;

	cout << endl
		<< "Params: " << predictability << "\t" << isolatedScalability << "\t" << occupiedScalability << endl
		<< endl;

	return output;
}

void thrdFunction(Operation* operation, Tensor* input, MyContext* context)
{
	context->queueOperation(operation);
	*input = operation->sequential->forward(*input);
	context->dequeueOperation();
}

void Operation::start(Tensor input)
{
	_output = &input;
	// _chosenContext = Scheduler::getMinimalContext(this);
	_chosenContext = Scheduler::getFastestContext(this);
	_th = thread(thrdFunction, this, &input, _chosenContext);
}

Tensor Operation::getResult()
{
	if (!_isException)
		_th.join();

	return *_output;
}

Tensor Operation::runSync(Tensor input)
{
	input = sequential->forward(input);
	_output = &input;
	return input;
}

void Operation::startSchedule(string name, Tensor input)
{
	auto now = steady_clock::now();

	if (true)//occupiedScalability < exceptionThreshold)
	{
		_isException = true;
		_chosenContext = Scheduler::selectDefaultContext();

		_chosenContext->select();
		_chosenContext->lock();
		_chosenContext->queueOperation(this);

		printfs("%s-->%s: started.\n", name.c_str(), _fullName.c_str());

		runSync(input);

		// printfs("%s-->%s: %3i SMs\t %i -> %li + %li = %li \n",
		// 	name.c_str(), _fullName.c_str(), _chosenContext->smCount,
		// 	(int)contextData[_chosenContext->index].occupiedExecutionTime,
		// 	duration_cast<microseconds>(steady_clock::now() - now).count(),
		// 	duration_cast<microseconds>(absoluteDeadline - steady_clock::now()).count(),
		// 	duration_cast<microseconds>(absoluteDeadline - now).count());

		_chosenContext->unlock();
		_chosenContext->release();
		_chosenContext->dequeueOperation();
	}

	else
	{
		_isException = false;
		start(input);
	}
}

Tensor Operation::scheduleSync(string name, Tensor input)
{
	// cout << "          Time: " << ((duration_cast<microseconds>(steady_clock::now().time_since_epoch())).count() % 1000000) << endl;
	// printfs("Starting     %s-->%s\n", name.c_str(), _fullName.c_str());
	// cout << "          STime of: " << name << "-->" << _fullName << " -> " << ((duration_cast<microseconds>(steady_clock::now().time_since_epoch())).count() % 1000000) << endl;

	if (occupiedScalability < exceptionThreshold)
	{
		_isException = true;
		_chosenContext = Scheduler::selectDefaultContext();
		_chosenContext->queueOperation(this);
		_chosenContext->lock();
	}

	else
	{
		_isException = false;
		_chosenContext = Scheduler::getMinimalContext(this);
		// _chosenContext = Scheduler::getFastestContext(this);
	}

	startTime = steady_clock::now();
	_chosenContext->select();

	auto now = startTime;
	// cout << "          Time: " << ((duration_cast<microseconds>(steady_clock::now().time_since_epoch())).count() % 1000000) << endl;
	// printfs("Executing     %s-->%s: %i: %li\n", name.c_str(), _fullName.c_str(), _chosenContext->smCount, _chosenContext->queue.size());

	// cout << "          CTime of: " << name << "-->" << _fullName << " -> " << ((duration_cast<microseconds>(steady_clock::now().time_since_epoch())).count() % 1000000) << endl;

	input = runSync(input);

	// cout << "          ETime of: " << name << "-->" << _fullName << " -> " << ((duration_cast<microseconds>(steady_clock::now().time_since_epoch())).count() % 1000000) << endl;

	// auto now2 = std::chrono::system_clock::now();
	// auto us = std::chrono::duration_cast<std::chrono::microseconds>(now2.time_since_epoch()).count();
	// std::time_t now_c = std::chrono::system_clock::to_time_t(now2);
	// char buffer[80];
	// std::strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", std::localtime(&now_c));
	// std::cout << buffer << "." << std::setfill('0') << std::setw(6) << (us % 1000000) << std::endl;

	// struct timespec ts;
	// clock_gettime(CLOCK_REALTIME, &ts);
	// time_t nowtime = ts.tv_sec;
	// struct tm* nowtm = localtime(&nowtime);
	// char tmbuf[64];
	// strftime(tmbuf, sizeof(tmbuf), "%Y-%m-%d %H:%M:%S", nowtm);
	// std::cout << "Current time: " << tmbuf << "." << ts.tv_nsec / 1000 << " microseconds" << std::endl;

	_parent->schedulerLogger->info(
		"FINISHED     {}-->{}: {} SMs\t {} -> {} + {} = {} ({})",
		name.c_str(), _fullName.c_str(), _chosenContext->smCount,
		(int)contextData[_chosenContext->index].occupiedExecutionTime,
		duration_cast<microseconds>(steady_clock::now() - now).count(),
		duration_cast<microseconds>(absoluteDeadline - steady_clock::now()).count(),
		duration_cast<microseconds>(absoluteDeadline - now).count(),
		(long)relativeDeadline[2]);

	// cout << name << "-->" << _fullName << " UNlocking: " << _chosenContext->smCount << endl;
	_chosenContext->unlock();
	_chosenContext->release();
	_chosenContext->dequeueOperation();

	return input;
}

double Operation::getRegulatedExecutionTime(int contextIndex)
{
	// printfs("Sca: %lf, Exe: %lf, Reg: %lf, Exp: %lf\n", occupiedScalability, contextData[contextIndex].occupiedExecutionTime, contextData[contextIndex].occupiedExecutionTime * (1 - occupiedScalability), exp(contextData[contextIndex].occupiedExecutionTime * (1 - occupiedScalability) - 10000));
	return contextData[contextIndex].occupiedExecutionTime;// *max(1 - occupiedScalability, 0.25);
}

void Operation::setAbsoluteDeadline(int level, steady_clock::time_point start)
{
	absoluteDeadline = start + microseconds((int)stackedDeadline[level - 1]);
	// cout << level << endl;
	// cout << getFullName() << "->" << stackedDeadline[level - 1] << endl;
}