# include <main.hpp>

# include <cnt.hpp>

# include <schd.hpp>
# include <ctxd.hpp>
# include "trc.hpp"

# include <torch/torch.h>

# include <iostream>
# include <chrono>
# include <thread>
# include <unistd.h>
# include <filesystem>
# include <memory>

# include <nvToolsExt.h>

using namespace torch;
using namespace torch::nn;

using namespace FGPRS;

using namespace std;
using namespace chrono;

MyContainer::MyContainer(const MyContainer& container) : Module(container), tracker(ModuleTracker(this)) {}

void MyContainer::setFrequency(int frequency)
{
	this->frequency = frequency;
	interval = 1000000 / frequency;
}

void MyContainer::assignExecutionTime()
{
	wret = 0;

	for (auto op : operations)
		wret += op->wret;
}

void MyContainer::updateExecutionTime()
{
	if (_iterationCount < ModuleTracker::windowSize)
		return;

	for (auto op : operations)
	{
		auto record = tracker.records.rbegin();
		int count = ModuleTracker::windowSize;
		op->wret = 0;

		while (count--)
		{
			if ((*record)->accepted && (*record)->operations[op->id]->executionTime > op->wret)
				op->wret = (*record)->operations[op->id]->executionTime;

			record++;
		}

		currentRecord->setOperationWret(op.get(), op->wret);
	}

	assignExecutionTime();
}

void MyContainer::updateUtilization()
{
	updateExecutionTime();

	utilizationIsolated = (double)bcet / interval;
	utilizationPartitioned = (double)wret / interval;// / (Scheduler::contextCount * MyContext::streamCount);
}

void MyContainer::addOperation(shared_ptr<Operation> operation)
{
	operations.push_back(operation);
	operation->id = operations.size() - 1;
	operationCount++;
}

void MyContainer::reset()
{
	for (auto op : operations)
	{
		op->released = false;
		op->finished = false;
		op->delayed = false;
		op->priorDelayed = false;
	}
}

void MyContainer::runDummy()
{
	_stopDummy = false;

	_dummyThread = thread([this]()
		{
			currentContext->select();
			auto str = currentContext->getStream();
			str->select();

			auto input = torch::randn({ 1, 3, 224, 224 }).cuda();
			int repCount = 0;
			auto start = steady_clock::now();

			while (!_stopDummy)
			{
				auto output = forwardDummy(input, str);
				str->synchronize();
				repCount++;
			}

			auto end = steady_clock::now();
			auto duration = duration_cast<microseconds>(end - start).count();
			auto repDuration = duration / repCount;
			str->release();
		});
}

void MyContainer::stopDummy()
{
	_stopDummy = true;
}

void MyContainer::waitDummy()
{
	_dummyThread.join();
}

void MyContainer::analyzeBCET(int warmup, int repeat)
{
	bcet = 0;
	Scheduler::selectDefaultContext()->select();
	auto input = torch::randn({ 1, 3, inputSize, inputSize }).cuda();

	for (auto op : operations)
	{
		input = op->analyzeBCET(warmup, repeat, input);
		bcet += op->bcet;
	}
}

void MyContainer::analyzeWCET(int warmup, int repeat)
{
	wcet = 0;
	currentContext->select();
	auto input = torch::randn({ 1, 3, inputSize, inputSize }).cuda();
	Scheduler::runDummies(operations[0]->container);
	this_thread::sleep_for(chrono::milliseconds(10));

	for (auto op : operations)
	{
		input = op->analyzeWCET(warmup, repeat, input);
		wcet += op->wcet;
	}

	Scheduler::stopDummies();
	Scheduler::waitDummies();
	wret = wcet;

	updateUtilization();
}

void MyContainer::assignDeadline()
{
	int deadlineStack = 0;

	for (auto op : operations)
	{
		op->relativeDeadline = (int)round((double)op->wret / wret * interval);
		deadlineStack += op->relativeDeadline;
		op->stackedDeadline = deadlineStack;
	}
}

void MyContainer::setAbsoluteDeadline(steady_clock::time_point start)
{
	_lastArrival = start;

	for (auto op : operations)
		op->absoluteDeadline = start + microseconds((int)op->stackedDeadline);
}

Tensor MyContainer::forwardDummy(Tensor input, MyStream* str)
{
	for (auto op : operations)
	{
		input = op->sequential->forward(input);
		str->synchronize();
	}

	return input;
}

bool MyContainer::doesMeetDeadline()
{
	return (currentContext->activeUtilization + utilizationPartitioned) < 1.0;
}

bool MyContainer::isFair()
{
	return (acceptanceRate < Scheduler::acceptanceRate * 1.25);
}

int MyContainer::admissionTest()
{
	if (!isFair())
		return -1;

	if (doesMeetDeadline())
		return 1;
	// return -3;
	double minUtilization = currentContext->activeUtilization;
	MyContext* context = nullptr;

	for (int i = 0; i < Scheduler::contextCount; i++)
	{
		if (Scheduler::contextPool[i].activeUtilization < minUtilization)
		{
			minUtilization = Scheduler::contextPool[i].activeUtilization;
			context = &Scheduler::contextPool[i];
		}
	}

	if (context == nullptr)
		return -2;

	if ((minUtilization + utilizationPartitioned) < 1.0)
	{
		cout << "Container " << moduleName << " moved from " << currentContext->index << " to " << context->index << endl
			<< "\tMod Utilization: " << utilizationPartitioned << endl
			<< "\tOld Utilization: " << currentContext->activeUtilization << endl
			<< "\tNew Utilization: " << context->activeUtilization << endl;

		currentContext->removeModule(operations[0]->container);
		currentContext = context;
		currentContext->assignModule(operations[0]->container);

		return 2;
	}

	return -2;
}

void MyContainer::updateAcceptanceRate(bool accepted)
{
	acceptedCount += accepted;
	missedCount += !accepted;
	acceptanceRate = (double)acceptedCount / (acceptedCount + missedCount);

	currentContext->acceptedCount += accepted;
	currentContext->missedCount += !accepted;
	currentContext->acceptanceRate =
		(double)currentContext->acceptedCount /
		(currentContext->acceptedCount + currentContext->missedCount);

	Scheduler::acceptedCount += accepted;
	Scheduler::missedCount += !accepted;
	Scheduler::acceptanceRate = (double)Scheduler::acceptedCount / (Scheduler::acceptedCount + Scheduler::missedCount);
}

bool MyContainer::release(Tensor input)
{
	auto accepted = true;
	reset();
	auto opPrev = operations.back();
	_iterationCount++;
	int testResult = admissionTest();
	testResult = doesMeetDeadline();
	accepted = highPriority || (testResult > 0);

	if (!highPriority)
		updateAcceptanceRate(accepted);

	if (!accepted)
	{
		cout << "Container " << moduleName << " skipped" << endl
			<< "\tTest Result: " << testResult << endl
			<< "\tContainer Acceptance Rate: " << acceptanceRate << endl
			<< "\tContext Acceptance Rate: " << currentContext->acceptanceRate << endl
			<< "\tScheduler Acceptance Rate: " << Scheduler::acceptanceRate << endl
			<< "\tModule Utilization: " << utilizationPartitioned << endl
			<< "\tContext Utilization: " << currentContext->activeUtilization << endl;
		auto record = make_shared<ModuleTrackingRecord>(this, false);
		tracker.addRecord(record);
		return false;
	}

	if (!highPriority)
	{
		cout << "Container " << moduleName << " passed" << endl
			<< "\tContainer Acceptance Rate: " << acceptanceRate << endl
			<< "\tContext Acceptance Rate: " << currentContext->acceptanceRate << endl
			<< "\tScheduler Acceptance Rate: " << Scheduler::acceptanceRate << endl;
	}

	active = true;
	currentContext->updateUtilization();
	currentContext->select();

	auto record = make_shared<ModuleTrackingRecord>(this, true);
	currentRecord = record;

	auto start = steady_clock::now();

	for (int i = 0; i < operations.size(); i++)
	{
		auto op = operations[i];
		op->priorDelayed = opPrev->delayed;
		input = op->releaseSync(input);
		opPrev = op;
	}

	auto end = steady_clock::now();
	auto duration = duration_cast<microseconds>(end - start).count();
	record->setResponseTime(duration);
	tracker.addRecord(record);
	auto temp = wret;
	updateUtilization();
	assignDeadline();
	active = false;
	currentContext->updateUtilization();

	if (testResult == -2)
	{
		cout << "Container " << moduleName << " moved" << endl
			<< "\tOld wret: " << temp << endl
			<< "\tNew wret: " << wret << endl;
	}

	// return input;
	return true;
}