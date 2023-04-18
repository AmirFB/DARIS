# include <opr.hpp>

# include <ctxd.hpp>
# include <schd.hpp>

# include <torch/torch.h>

# include <chrono>
# include <iostream>
# include <unistd.h>
# include <future>
# include <cmath>

using namespace std;
using namespace chrono;
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
	NoGradGuard no_grad;
	Tensor output;
	bool first = true;
	steady_clock::time_point t1, t2, tStart, tEnd, tNow;
	int countIsolated, countOccupied;

	predictability = 0;
	isolatedScalability = 0;
	occupiedScalability = 0;

	_parent->analyzeLogger->info("{}:", _fullName);
	// cout << _fullName << ":" << endl;

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
		_parent->analyzeLogger->info("\t{}\t{:.0f}us, {:.0f}us",
			ctx->smCount, contextData.back().isolatedExecutionTime, contextData.back().occupiedExecutionTime);

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

	_parent->analyzeLogger->info("Params: {:.2f}\t{:.2f}\t{:.2f}\n", predictability, isolatedScalability, occupiedScalability);

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

void Operation::startSchedule(Tensor input)
{
	auto now = steady_clock::now();

	if (true)//occupiedScalability < exceptionThreshold)
	{
		_isException = true;
		_chosenContext = Scheduler::selectDefaultContext();

		_chosenContext->select();
		_chosenContext->lock();
		_chosenContext->queueOperation(this);

		// printf("%s-->%s: started.\n", name.c_str(), _fullName.c_str());

		runSync(input);

		// printf("%s-->%s: %3i SMs\t %i -> %li + %li = %li \n",
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

Tensor Operation::scheduleSync(Tensor input)
{
	// cout << "          Time: " << ((duration_cast<microseconds>(steady_clock::now().time_since_epoch())).count() % 1000000) << endl;
	// printf("Starting     %s-->%s\n", name.c_str(), _fullName.c_str());
	// cout << "          STime of: " << name << "-->" << _fullName << " -> " << ((duration_cast<microseconds>(steady_clock::now().time_since_epoch())).count() % 1000000) << endl;

	_parent->scheduleLogger->info("");

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
		// _chosenContext = Scheduler::getMinimalContext(this);
		_chosenContext = Scheduler::getFastestContext(this);
	}

	startTime = steady_clock::now();
	_chosenContext->select();

	auto now = startTime;

	_parent->scheduleLogger->info("Start  {}: {} SMs -> {}",
		_fullName.c_str(), _chosenContext->smCount, _chosenContext->queue.size());

	input = runSync(input);

	_parent->scheduleLogger->info(
		"End    {}: {} -> {} + {} = {} ({})",
		_fullName.c_str(),
		(int)contextData[_chosenContext->index].occupiedExecutionTime,
		duration_cast<microseconds>(steady_clock::now() - now).count(),
		duration_cast<microseconds>(absoluteDeadline - steady_clock::now()).count(),
		duration_cast<microseconds>(absoluteDeadline - now).count(),
		(long)relativeDeadline[2]);
	_parent->scheduleLogger->info("");

	_chosenContext->unlock();
	_chosenContext->release();
	_chosenContext->dequeueOperation();

	return input;
}

double Operation::getRegulatedExecutionTime(int contextIndex)
{
	// printf("Sca: %lf, Exe: %lf, Reg: %lf, Exp: %lf\n", occupiedScalability, contextData[contextIndex].occupiedExecutionTime, contextData[contextIndex].occupiedExecutionTime * (1 - occupiedScalability), exp(contextData[contextIndex].occupiedExecutionTime * (1 - occupiedScalability) - 10000));
	return contextData[contextIndex].occupiedExecutionTime;// *max(1 - occupiedScalability, 0.25);
}

void Operation::setAbsoluteDeadline(int level, steady_clock::time_point start)
{
	absoluteDeadline = start + microseconds((int)stackedDeadline[level - 1]);
	// cout << level << endl;
	// cout << getFullName() << "->" << stackedDeadline[level - 1] << endl;
}