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

ParalleltialImpl::ParalleltialImpl(vector<shared_ptr<SequentialImpl>> parallels) : _parallels(parallels) {}

Tensor ParalleltialImpl::forward(Tensor input)
{
	vector<Tensor> outputs;

	for (auto parallel : _parallels)
		outputs.push_back(parallel->forward(input));

	Tensor output = cat(outputs, 1);
	return output;
}

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
	// if (_iterationCount < ModuleTracker::windowSize)
	if (tracker.records.size() == 0)
		return;

	for (auto op : operations)
	{
		auto record = tracker.records.rbegin();
		int count = min(ModuleTracker::windowSize, (int)tracker.records.size());
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
	utilizationPartitioned = (double)wret / interval;
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
	isDummy = true;
	_stopDummy = false;

	currentContext->select();
	auto str = currentContext->getStream();
	str->select();

	auto input = torch::randn({ 1, 3, inputSize, inputSize }).cuda();
	auto output = forwardDummy(input, str);
	str->release();
	isDummy = false;
}

void MyContainer::stopDummy()
{
	_stopDummy = true;
}

void MyContainer::waitDummy()
{
	_stopDummy = true;
	_dummyThread.join();
}

void MyContainer::analyzeOperations(int warmup, int repeat, bool isWcet)
{
	auto input = torch::randn({ 1, 3, inputSize, inputSize }).cuda();
	int timer;

	for (auto op : operations)
	{
		input = op->analyze(warmup, repeat, input, &timer);
		isWcet ? (op->wcet = timer, wcet += timer) : (op->bcet = timer, bcet += timer);
	}
}

void MyContainer::analyzeBCET(int warmup, int repeat)
{
	bcet = 0;
	currentContext->select();
	analyzeOperations(warmup, repeat, false);
}

void MyContainer::analyzeWCET(int warmup, int repeat)
{
	wcet = 0;
	currentContext->select();
	Scheduler::runDummies(operations[0]->container);
	this_thread::sleep_for(chrono::milliseconds(10));

	analyzeOperations(warmup, repeat, true);

	for (auto op : operations)
		op->wret = op->wcet;

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
	return (Scheduler::acceptanceRate == 0) || (acceptanceRate < Scheduler::acceptanceRate * 1.5);
}

int MyContainer::admissionTest()
{
	double minUtilization = 1000;
	MyContext* context = nullptr;

	if (highPriority)
		return 100;

	else if (_iterationCount > ModuleTracker::windowSize && acceptanceRate < (Scheduler::acceptanceRate * 0.5))
	{
		for (int i = 0; i < Scheduler::contextCount; i++)
		{
			if ((Scheduler::contextPool[i].activeUtilization < minUtilization) &&
				(Scheduler::contextPool[i].overallUtilization < (currentContext->overallUtilization - utilizationPartitioned * 0.99)))
			{
				minUtilization = Scheduler::contextPool[i].activeUtilization;
				context = &Scheduler::contextPool[i];
			}
		}

		if (context == nullptr)
			return -10;

		// if ((minUtilization + utilizationPartitioned) < 1.1)
		{
			cout << "Dontainer " << moduleName << " moved from " << currentContext->index << " to " << context->index << endl
				<< "\tMod Utilization: " << utilizationPartitioned << endl
				<< "\tOld Utilization: " << currentContext->activeUtilization << endl
				<< "\tNew Utilization: " << context->activeUtilization << endl;

			currentContext->removeModule(operations[0]->container);
			currentContext = context;
			currentContext->assignModule(operations[0]->container);

			return 10;
		}

		return -20;
	}

	if (!isFair())
		return -1;

	if (doesMeetDeadline())
		return 1;

	minUtilization = currentContext->activeUtilization;
	context = nullptr;

	for (int i = 0; i < Scheduler::contextCount; i++)
	{
		if ((Scheduler::contextPool[i].activeUtilization < minUtilization) &&
			(Scheduler::contextPool[i].overallUtilization < (currentContext->overallUtilization - utilizationPartitioned)))
		{
			minUtilization = Scheduler::contextPool[i].activeUtilization;
			context = &Scheduler::contextPool[i];
		}
	}

	if (context == nullptr)
		return -2;

	if ((minUtilization + utilizationPartitioned) < 1.0)
	{
		cout << "Eontainer " << moduleName << " moved from " << currentContext->index << " to " << context->index << endl
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
	skippedCount += !accepted;
	acceptanceRate = (double)acceptedCount / (acceptedCount + skippedCount);

	currentContext->acceptedCount += accepted;
	currentContext->skippedCount += !accepted;
	currentContext->acceptanceRate =
		(double)currentContext->acceptedCount /
		(currentContext->acceptedCount + currentContext->skippedCount);

	Scheduler::acceptedCount += accepted;
	Scheduler::skippedCount += !accepted;
	Scheduler::acceptanceRate = (double)Scheduler::acceptedCount / (Scheduler::acceptedCount + Scheduler::skippedCount);
}

Tensor MyContainer::releaseOperations(Tensor input)
{
	auto opPrev = operations.back();

	for (int i = 0; i < operations.size(); i++)
	{
		auto op = operations[i];
		op->priorDelayed = opPrev->delayed;
		input = op->releaseSync(input);
		opPrev = op;
	}

	return input;
}

bool MyContainer::release(Tensor input)
{
	auto accepted = true;
	reset();
	_iterationCount++;
	int testResult = highPriority ? 1000 : admissionTest();

	accepted = testResult > 0;

	if (!highPriority)
		updateAcceptanceRate(accepted);

	if (!accepted)
	{
		// cout << "Container " << moduleName << " skipped" << endl;
		// << "\tTest Result: " << testResult << endl
		// 	<< "\tContainer Acceptance Rate: " << acceptanceRate << endl
		// 	<< "\tContext Acceptance Rate: " << currentContext->acceptanceRate << endl
		// 	<< "\tScheduler Acceptance Rate: " << Scheduler::acceptanceRate << endl
		// 	<< "\tModule Utilization: " << utilizationPartitioned << endl
		// 	<< "\tContext Utilization: " << currentContext->activeUtilization << endl;
		auto record = make_shared<ModuleTrackingRecord>(this, false);
		tracker.addRecord(record);
		return false;
	}

	active = true;
	currentContext->updateUtilization();
	currentContext->select();

	auto record = make_shared<ModuleTrackingRecord>(this, true);
	currentRecord = record;

	auto start = steady_clock::now();
	releaseOperations(input);
	auto end = steady_clock::now();

	auto duration = duration_cast<microseconds>(end - start).count();
	record->setResponseTime(duration);
	tracker.addRecord(record);
	auto temp = wret;
	updateUtilization();
	assignDeadline();
	active = false;
	currentContext->updateUtilization();

	return true;
}