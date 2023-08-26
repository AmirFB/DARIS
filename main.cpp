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
// # include <random>
# include <ctime>
# include <filesystem>
// # include "spdlog/spdlog.h"
# include "spdlog/sinks/basic_file_sink.h"

# include <torch/torch.h>
# include <torch/script.h>
# include <c10/cuda/CUDAStream.h>
# include <ATen/cuda/CUDAContext.h>
# include <c10/cuda/CUDACachingAllocator.h>

# include <cuda.h>
# include <cudaTypedefs.h>
# include <cuda_runtime.h>
# include <cuda_profiler_api.h>
# include <nvToolsExt.h>

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
# include <deeplab.hpp>

# include <c10/cuda/CUDACachingAllocator.h>

using namespace std;
using namespace chrono;
using namespace spdlog;
using namespace torch;
using namespace torch::nn;
using namespace FGPRS;

# define PRELIMINNARY 0
# define SCHEDULER 		1
# define MODE					SCHEDULER

shared_ptr<logger> logger;

void distributeSMs(int* array, int total, int count);
vector<double> generateUtilization(int count, double total);

double maxFPS[] = { 769, 302, 92, 473, 144, 44 };

int main(int argc, char** argv)
{
# if MODE == SCHEDULER
	// srand(time(nullptr));
	auto logger = spdlog::basic_logger_mt("main_logger", "log.log");
	logger->set_pattern("[%S.%f] %v");
	NoGradGuard no_grad;
	int level = 2;
	int moduleCount, dummyCount = 0;
	double frequency;

	SchedulerType type;
	int* smOptions;
	int smCount;

	moduleCount = atoi(argv[2]);
	dummyCount = atoi(argv[3]);
	frequency = atof(argv[4]);

	cout << "Initializing scheduler ..." << endl;

	if (!strcmp(argv[1], "proposed"))
	{
		type = PROPOSED_SCHEDULER;

		smCount = atoi(argv[5]);
		smOptions = new int[smCount];

		for (int i = 0; i < smCount; i++)
			smOptions[i] = atoi(argv[6 + i]);
	}

	else if (!strcmp(argv[1], "mps"))
	{
		type = MPS_SCHEDULER;
		smCount = moduleCount;
		smOptions = new int[smCount];

		for (int i = 0; i < smCount; i++)
			smOptions[i] = 68;
	}

	else if (!strcmp(argv[1], "pmps"))
	{
		type = PMPS_SCHEDULER;
		smCount = moduleCount;
		smOptions = new int[smCount];
		distributeSMs(smOptions, 68, smCount);
	}

	else if (!strcmp(argv[1], "pmpso"))
	{
		type = PMPSO_SCHEDULER;
		smCount = moduleCount;
		smOptions = new int[smCount];
		distributeSMs(smOptions, max(68 * (smCount / 2 + 0.5), 68.0), smCount);
	}

	else if (!strcmp(argv[1], "nomps"))
	{
		type = NOMPS_SCHEDULER;
		smCount = 1;
		smOptions = new int[1] {68};
	}

	Scheduler::initialize(smOptions, smCount, type, true);
	MyContext::selectDefault();

	Tensor inputs[moduleCount];
	shared_ptr<MyContainer> mods[moduleCount];
	// shared_ptr<ResNet<BasicBlock>> mods[moduleCount];
	// shared_ptr<DeepLabV3Plus> mods[moduleCount];
	Loop loops[moduleCount];

	cout << "Initializing modules ..." << endl;
	filesystem::remove_all("logs");

	string name;
	int modIndex, inputSize = 224;
	double freq;
	// auto quotas = generateUtilization(moduleCount, frequency);
	string freqStr;

	// cudaProfilerStart();

	// vector<DummyContainer> dummySet;

	for (size_t i = 0; i < moduleCount; i++)
		Scheduler::dummyContainer.push_back(DummyContainer{ mods[i], &inputs[i], i });

	// random_shuffle(dummySet.begin(), dummySet.end());
	Scheduler::dummyContainer.resize(dummyCount + 1);
	// Scheduler::dummyContainer = vector<shared_ptr<DummyContainer>>(dummySet.begin(), dummySet.end());

	for (int i = 0; i < moduleCount; i++)
	{
		modIndex = rand() % 1;
		// freq = quotas[i] * maxFPS[modIndex];
		// freq = (i + 1) * 15 - 1;
		// freq = 200;

		freq = 30;

		stringstream stream;
		stream << fixed << setprecision(2) << freq;
		freqStr = stream.str();

		name = (modIndex < 3 ? "resnet" : "deeplab") +
			to_string(i + 1);// +"_" + to_string(inputSize) + "_" + freqStr + "Hz";

		// cout << "\t" << setprecision(2) << name << (i + 1) << " (" << (quotas[i] / frequency * 100) << "%)" << endl;

		// inputs[i] = torch::randn({ 1, 3, 224, 224 }, kCUDA);
		// mods[i] = resnet18(1000);
		// mods[i] = make_shared<DeepLabV3PlusImpl>(DeepLabV3PlusImpl(100, "resnet18"));
		// loops[i] = Loop(name + to_string(i + 1), mods[i], frequency, i);

		inputs[i] = torch::randn({ 1, 3, inputSize, inputSize }, kCUDA);

		if (modIndex < 3)
			mods[i] = resnet18(1000);
		else
			mods[i] = make_shared<DeepLabV3PlusImpl>(DeepLabV3PlusImpl(100, "resnet18"));

		loops[i] = Loop(name, mods[i], freq, i);
		loops[i].initialize(smCount - 1, inputs[i], type, level);
	}

	// cudaProfilerStop();

	// inputs[0] = torch::randn({ 1, 3, 1024, 1024 }, kCUDA);
	// mods[0] = resnet18(1000);
	// Loops[0] = Loop("res1", mods[0], 30, 0);
	// Loops[0].initialize(3, inputs[0], type, level);

	cout << "Warming up ..." << endl;

	// cudaProfilerStart();

	for (int i = 0; i < moduleCount; i++)
		loops[i].start(&inputs[i], type, level, false);

	this_thread::sleep_for(milliseconds(500));

	for (int i = 0; i < moduleCount; i++)
		loops[i].stop();

	for (int i = 0; i < moduleCount; i++)
		loops[i].wait();

	cout << endl << endl << endl << endl << endl;

	// system("clear");
	cout << "Memory: " << Scheduler::getFreeMemoryGB() << " GB" << endl;
	cout << "Here we go ...\n";

	auto now = std::time(nullptr);
	char time_string[20];
	std::strftime(time_string, sizeof(time_string), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
	std::cout << "Current time: " << time_string << ".";
	std::clock_t clock = std::clock();
	double microseconds = 1000000.0 * static_cast<double>(clock) / CLOCKS_PER_SEC;
	std::cout << std::fixed << std::setprecision(0) << std::setw(6) << std::setfill('0') << microseconds << std::endl;

	logger->info("Started!");

	cudaProfilerStart();
	nvtxRangePush("whole");

	for (int i = 0; i < moduleCount; i++)
		loops[i].start(&inputs[i], type, level, true);

	this_thread::sleep_for(milliseconds(1000));

	for (int i = 0; i < moduleCount; i++)
		loops[i].stop();

	for (int i = 0; i < moduleCount; i++)
		loops[i].wait();

	nvtxRangePop();
	cudaProfilerStop();

	logger->info("Finished!");

# elif MODE == PRELIMINARY

	char* op = argv[1];
	mkdir("results", 0777);

	if (!strcmp(op, "clear"))
	{
		cout << "Removing previous results of \"" << argv[2] << "\" simulation\n";

		if (!strcmp(op, "concurrency"))
		{
			remove(string("results/concurrency1.csv").c_str());
			remove(string("results/concurrency2.csv").c_str());
			remove(string("results/concurrency3.csv").c_str());
			remove(string("results/concurrency4.csv").c_str());
			remove(string("results/concurrency5.csv").c_str());
		}

		else
			remove((string("results/") + string(argv[2]) + ".csv").c_str());
	}

	else if (!strcmp(op, "speedup"))
		testSpeedup(&argv[2]);

	else if (!strcmp(op, "concurrency"))
		testConcurrency(&argv[2]);

# endif
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

vector<double> generateUtilization(int count, double total)
{
	// vector<double> result(count);
	// random_device rd;
	// std::mt19937 gen(rd());
	// uniform_real_distribution<double> dis(0.05, 0.9);

	// double sum = 0;

	// for (int i = 0; i < count; i++)
	// {
	// 	result[i] = dis(gen);
	// 	sum += result[i];
	// }

	// for (int i = 0; i < count; i++)
	// 	result[i] = result[i] / sum * total;

	// return result;
}