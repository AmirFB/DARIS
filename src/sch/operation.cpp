# include <torch/torch.h>
# include "operation.h"


# include <chrono>
# include <iostream>

using namespace std;
using namespace std::chrono;
using namespace FGPRS;

Operation::Operation(AnyModule *module)
{
	_module = module;
	_isSequential = true;
}

Operation::Operation(Sequential *sequence)
{
	_sequence = sequence;
	_isSequential = false;
}

Tensor* Operation::forward(Tensor *input)
{
	if (!_isSequential)
		*_output = _module->forward(*input);
	
	else
		*_output = (*_sequence)->forward(*input);
}

void Operation::initialize(Tensor *dummyInput, int warmup, int repeat)
{
	Tensor output;
	int sm;
	bool result;
	MyContext *ctx;
	typedef high_resolution_clock Clock;

	_execTime = (double*)malloc(sizeof(double) * Scheduler::maxSmCount);

	for (int i = 0; i < Scheduler::poolSize; i++)
	{
		sm = Scheduler::smOptions[i];

		ctx = Scheduler::selectContext(sm);
		cout << "Debug: " << sm << ", " << ctx->select(0) << endl;
		ctx->select(0);

		for (int j = 0; j < warmup; j++)
			forward(dummyInput);

		auto t1 = Clock::now();

		for (int j = 0; j < repeat; j++)
			forward(dummyInput);

		auto t2 = Clock::now();

		std::chrono::duration<double> d;
		d = t2 - t1;
		_execTime[sm] = d.count() / repeat;

		cout << "SM: " << sm << "\tTime: " << _execTime[sm] * 1000000 << endl;
	}
}