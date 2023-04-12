
# include <cnt.hpp>

# include <schd.hpp>
# include <ctxd.hpp>

# include <torch/torch.h>

# include <iostream>
# include <chrono>
# include <thread>
# include <unistd.h>

using namespace torch;
using namespace torch::nn;

using namespace FGPRS;

using namespace std;
using namespace std::chrono;

vector<shared_ptr<Operation>> MyContainer::getOperations(int level)
{
	return operations[level - 1];
}

void MyContainer::copyOperations(string parentName, MyContainer& container, int level)
{
	for (auto op : container.getOperations(1))
	{
		op->setParentName(parentName);
		operations[0].push_back(op);
	}

	if (level != 2)
	{
		for (auto op : container.getOperations(2))
		{
			op->setParentName(parentName);
			operations[1].push_back(op);
		}
	}

	if (level == 1)
	{
		for (auto op : container.getOperations(3))
		{
			op->setParentName(parentName);
			operations[2].push_back(op);
		}
	}
}

Tensor MyContainer::schedule(string name, Tensor input, int level)
{
	for (auto op : operations[level - 1])
		input = op->scheduleSync(name, input);

	return input;
}

void MyContainer::analyze(int warmup, int repeat, Tensor input)
{
	for (int i = 1; i <= 3; i++)
		analyze(warmup, repeat, input, i);
}

Tensor MyContainer::analyze(int warmup, int repeat, Tensor input, int level)
{
	for (auto op : operations[level - 1])
		input = op->analyze(warmup, repeat, input);

	return input;
}

double MyContainer::assignExecutionTime(int level, int contextIndex, double executionTimeStack)
{
	double timeStack = 0;
	level -= 1;

	contextData[level].resize(Scheduler::smOptions.size());

	for (int i = 0; i < Scheduler::smOptions.size(); i++)
	{
		contextData[level][i] = ContextData(Scheduler::selectContextByIndex(i));

		for (auto op : operations[level])
			contextData[level][i].stackExecutionTime(op->contextData[i]);
	}

	for (auto op : operations[level])
		timeStack += op->getRegulatedExecutionTime(contextIndex);

	regulatedExecutionTime[level] = timeStack;
	return executionTimeStack + timeStack;
}

double MyContainer::assignDeadline(double quota, int level, int contextIndex, double deadlineStack)
{
	level -= 1;

	for (auto op : operations[level])
	{
		op->relativeDeadline[level] = op->getRegulatedExecutionTime(contextIndex) / regulatedExecutionTime[level] * quota;
		deadlineStack += op->relativeDeadline[level];
		op->stackedDeadline[level] = deadlineStack;

		// cout << op->getFullName() << ": " << op->relativeDeadline[level] << "-->" << op->stackedDeadline[level] << endl;
	}

	return deadlineStack;
}

void MyContainer::setAbsoluteDeadline(int level, steady_clock::time_point start)
{
	for (auto op : operations[level - 1])
		op->setAbsoluteDeadline(level, start);
}