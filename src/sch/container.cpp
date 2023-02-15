#include <mod.h>

#include <schd.h>
#include <ctx.h>

#include <iostream>
#include <chrono>
#include <thread>
#include <torch/torch.h>

using namespace torch;
using namespace torch::nn;

using namespace FGPRS;

using namespace std;
using namespace std::chrono;

Tensor MyModule::analyze(int warmup, int repeat, Tensor dummyData, vector<int> smOptions)
{
	cout << "Let's analyze: " << _operations.size() << endl;

	for (auto op : _operations)
	{
		if (op.getName() == "fc")
			dummyData = flatten(dummyData, 1);

		dummyData = op.analyze(warmup, repeat, dummyData, smOptions);
	}

	return dummyData;
}

vector<Operation> MyModule::getOperations()
{
	return _operations;
}

void MyModule::addOperations(vector<Operation> operations)
{
	_operations.insert(_operations.end(), operations.begin(), operations.end());
}

void MyModule::addOperations(string parentName, vector<Operation> operations)
{
	for (auto op : operations)
	{
		op.setParentName(parentName);
		_operations.push_back(op);
	}
}