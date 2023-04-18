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
# include "spdlog/spdlog.h"
# include "spdlog/sinks/basic_file_sink.h"

# include <torch/torch.h>
# include <torch/script.h>
# include <c10/cuda/CUDAStream.h>
# include <ATen/cuda/CUDAContext.h>
# include <c10/cuda/CUDACachingAllocator.h>

# include <cuda.h>
# include <cudaTypedefs.h>
# include <cuda_runtime.h>

# include <loop.hpp>
# include <schd.hpp>
# include <cnt.hpp>

# include <cif10.hpp>
# include <resnet.hpp>

# include <tests.hpp>

# include <stdio.h>
# include <stdlib.h>
# include <unistd.h>
# include <sys/time.h>
# include <sched.h>

# include <c10/cuda/CUDACachingAllocator.h>

using namespace std;
using namespace chrono;
using namespace spdlog;
using namespace torch;
using namespace torch::nn;
using namespace FGPRS;

# define MODULES_COUNT 35
# define REPEAT 1

shared_ptr<logger> logger;

void distributeSMs(int* array, int total, int count);

int main(int argc, char** argv)
{
	auto logger = spdlog::basic_logger_mt("main_logger", "log.log");
	logger->set_pattern("[%S.%f] %v");
	NoGradGuard no_grad;
	int level = 3;

	SchedulerType type;
	int* smOptions;
	int smCount;

	if (!strcmp(argv[1], "proposed"))
	{
		type = PROPOSED_SCHEDULER;
		smCount = 4;
		smOptions = new int[] { 10, 20, 30, 40 };
	}

	else if (!strcmp(argv[1], "mps"))
	{
		type = MPS_SCHEDULER;
		smCount = MODULES_COUNT;
		smOptions = new int[smCount];

		for (int i = 0; i < smCount; i++)
			smOptions[i] = 68;
	}

	else if (!strcmp(argv[1], "pmps"))
	{
		type = PMPS_SCHEDULER;
		smCount = MODULES_COUNT;
		smOptions = new int[smCount];
		distributeSMs(smOptions, 68, smCount);
	}

	else if (!strcmp(argv[1], "pmpso"))
	{
		type = PMPSO_SCHEDULER;
		smCount = MODULES_COUNT;
		smOptions = new int[smCount];
		distributeSMs(smOptions, 68 * (smCount / 2 + 0.5), smCount);
	}

	else if (!strcmp(argv[1], "nomps"))
	{
		type = NOMPS_SCHEDULER;
		smCount = 1;
		smOptions = new int[] {68};
	}

	Scheduler::initialize(smOptions, smCount, type);

	Tensor inputs[MODULES_COUNT];
	shared_ptr<ResNet<BasicBlock>> mods[MODULES_COUNT];
	Loop loops[MODULES_COUNT];

	for (int i = 0; i < MODULES_COUNT; i++)
	{
		string name = "res";

		inputs[i] = torch::randn({ 1, 3, 224, 224 }, kCUDA);
		mods[i] = resnet18(1000);
		loops[i] = Loop(name + to_string(i + 1), mods[i], 20, i);
		loops[i].initialize(3, inputs[i], type, level);
	}

	for (int i = 0; i < MODULES_COUNT; i++)
		loops[i].start(&inputs[i], type, level);

	this_thread::sleep_for(milliseconds(250));

	for (int i = 0; i < MODULES_COUNT; i++)
		loops[i].stop();

	cout << endl << endl << endl << endl << endl;

	system("clear");
	cout << "Here we go ...\n";

	logger->info("Started!");

	for (int j = 0; j < REPEAT; j++)
	{
		for (int i = 0; i < MODULES_COUNT; i++)
			loops[i].start(&inputs[i], type, level);

		this_thread::sleep_for(milliseconds(1000));

		for (int i = 0; i < MODULES_COUNT; i++)
			loops[i].stop();

		cout << endl << endl << endl << endl << endl;
	}

	logger->info("Finished!");

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

void distributeSMs(int* array, int total, int count)
{
	int minPer, rem;

	minPer = total / count - (total / count) % 2;
	rem = total - count * minPer;

	for (int i = 0; i < count; i++)
	{
		array[i] = minPer;

		if (rem > 0)
		{
			array[i] += 2;
			rem -= 2;
		}
	}
}