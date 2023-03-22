# include <iostream>
# include <fstream>
# include <iomanip>
# include <thread>
# include <pthread.h>
# include <chrono>
# include <string>
# include <cstdlib>
# include <future>
# include <sys/stat.h>

# include <torch/torch.h>
# include <torch/script.h>
# include <c10/cuda/CUDAStream.h>
# include <ATen/cuda/CUDAContext.h>
# include <c10/cuda/CUDACachingAllocator.h>

# include <cuda.h>
# include <cudaTypedefs.h>
# include <cuda_runtime.h>

# define SCHEDULER_TYPE	PROPOSED

# include <schd.hpp>
# include <cnt.hpp>
# include <loop.hpp>

# include <cif10.hpp>
# include <resnet.hpp>

# include <tests.hpp>

# include <stdio.h>
# include <stdlib.h>
# include <unistd.h>

# include <c10/cuda/CUDACachingAllocator.h>

using namespace std;
using namespace std::chrono;
using namespace torch;
using namespace torch::nn;
using namespace FGPRS;

# define COUNT 5
# define REPEAT 10

int main(int argc, char** argv)
{
	NoGradGuard no_grad;
	int level = 3;

	int smOptions[] = { 6, 12, 22, 44 };
	Scheduler::initialize(smOptions, 4);

	Tensor inputs[COUNT];
	shared_ptr<ResNet<BasicBlock>> mods[COUNT];
	Loop loops[COUNT];

	for (int i = 0; i < COUNT; i++)
	{
		inputs[i] = torch::randn({ 1, 3, 224, 224 }, kCUDA);
		mods[i] = resnet18(1000);
		loops[i] = Loop("res" + to_string(i + 1), mods[i], 85);
		loops[i].initialize(level, inputs[i]);
	}

	for (int i = 0; i < COUNT; i++)
		loops[i].start(&inputs[i], level);

	this_thread::sleep_for(milliseconds(10));

	for (int i = 0; i < COUNT; i++)
		loops[i].stop();

	cout << endl << endl << endl << endl << endl;

	for (int j = 0; j < REPEAT; j++)
	{
		this_thread::sleep_for(milliseconds(100));

		for (int i = 0; i < COUNT; i++)
			loops[i].start(&inputs[i], level);

		this_thread::sleep_for(milliseconds(200));

		for (int i = 0; i < COUNT; i++)
			loops[i].stop();

		cout << endl << endl << endl << endl << endl;
	}

	// char *op = argv[1];
	// mkdir("results", 0777 );

	// if (!strcmp(op, "clear"))
	// {
	// 	cout << "Removing previous results of \"" << argv[2] << "\" simulation\n";
	// 	remove((string("results/") + string(argv[2]) + ".csv").c_str());
	// }

	// else if (!strcmp(op, "speedup"))
	// 	testSpeedup(&argv[2]);

	// // else if (!strcmp(op, "concurrency"))
	// // 	testConcurrency(&argv[2]);

	// else if (!strcmp(op, "tailing"))
	// 	testTailing(&argv[2]);

	// else if (!strcmp(op, "interference"))
	// 	testInterference(&argv[2]);
}