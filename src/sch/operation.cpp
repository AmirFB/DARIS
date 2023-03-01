# include <mod.h>

# include <schd.h>
# include <ctx.h>

# include <torch/torch.h>

# include <chrono>
# include <iostream>
# include <unistd.h>
# include <future>

using namespace std;
using namespace std::chrono;
using namespace FGPRS;

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

	while (true)
	{
		output = _sequential->forward(input);

		if (tEnd <= steady_clock::now())
			break;
	}

	for (auto sm : Scheduler::smOptions)
	{
		auto ctx = Scheduler::selectContext(sm);
		ctx->select();

		tStart = steady_clock::now();
		tEnd = tStart + milliseconds(warmup);

		while (true)
		{
			output = _sequential->forward(input);

			if (tEnd <= steady_clock::now())
				break;
		}

		countIsolated = 0;
		tStart = steady_clock::now();
		tEnd = tStart + milliseconds(repeat);

		while (true)
		{
			output = _sequential->forward(input);
			countIsolated++;
			tNow = steady_clock::now();

			if (tEnd <= tNow)
				break;
		}

		duration<double> d1 = tNow - tStart;

		ctx->release();
		Scheduler::startDummy(ctx);
		usleep(1000);
		ctx->select();

		countOccupied = 0;
		tStart = steady_clock::now();
		tEnd = tStart + milliseconds(repeat);

		while (true)
		{
			output = _sequential->forward(input);
			countOccupied++;
			tNow = steady_clock::now();

			if (tEnd <= tNow)
				break;
		}

		duration<double> d2 = tNow - tStart;

		Scheduler::stopDummy();
		ctx->release();

		contextData.push_back(ContextData(ctx, d1.count() / countIsolated * 1000000, d2.count() / countOccupied * 1000000));
		cout << "\t" << ctx->smCount << "\t" << contextData.back().isolatedExecutionTime << "us"
			<< ", " << contextData.back().occupiedExecutionTime << "us";

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

void thrdFunction(Sequential* sequential, Tensor* input)
{
	*input = (*sequential)->forward(*input);
}

void Operation::start(Tensor input)
{
	auto _sync = async(launch::async, thrdFunction, &_sequential, &input);
}

Tensor Operation::getResult()
{
	return _pAsync->get();
}

Tensor Operation::runAsync(Tensor input)
{
	auto th = async(launch::async, thrdFunction, &_sequential, &input);
	th.get();
	return input;
}

Tensor Operation::runSync(Tensor input)
{
	return _sequential->forward(input);
}

Tensor Operation::runThread(Tensor input)
{
	auto th = thread(thrdFunction, &_sequential, &input);
	th.join();
	return input;
}

double Operation::getRegulatedExecutionTime(int contextIndex)
{
	return contextData[contextIndex].occupiedExecutionTime * (1 - occupiedScalability);
}