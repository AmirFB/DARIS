# include <iostream>
# include <vector>
# include <cuda_profiler_api.h>
# include <torch/script.h>

# include <libsmctrl.h>

# include "scenario.hpp"
# include "cnt.hpp"
# include "schd.hpp"
# include "loop.hpp"

using namespace std;
using namespace FGPRS;

int type, taskset;

int contextCount, smCount, streamCount;
double oversubscription;
int resnetLowCount, resnetHighCount, unetLowCount, unetHighCount, inceptionLowCount, inceptionHighCount;
vector<shared_ptr<Loop>> loops;
int timer, windowSize;
int maxBatchSize;

int warmup, repeat;
const int frequency = 24, inputSize = 224;

Scenario scenario;

int main()
{
	MyContext::mainStreamCount = 1;
	MyContext::secondaryStreamCount = 0;
	Scheduler::initialize(1, 8);

	auto ctx = Scheduler::selectContextByIndex(0);
	ctx->select();
	auto str = ctx->getStream();
	auto stream = str->stream();

	str->select();
	torch::Device device(torch::kCUDA);

	// Load the TorchScript model from the ZIP archive with CUDA support
	std::string model_path = "resnet.zip";
	torch::jit::script::Module model = torch::jit::load(model_path, device);

	// Optionally, set the model to evaluation mode
	model.eval();

	// Generate example input data
	auto inputs = torch::randn({ 4, 3, 224, 224 }).to(device);

	// Perform inference
	auto output = model.forward({ inputs.to(device) }).toTensor();
	str->synchronize();

	for (int i = 0; i < 10; i++)
	{
		output = model.forward({ inputs }).toTensor();
		str->synchronize();
	}

	auto start = chrono::high_resolution_clock::now();

	for (int i = 0; i < 100; i++)
	{
		output = model.forward({ inputs }).toTensor();
		str->synchronize();
	}

	auto end = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(end - start).count();

	cout << "Duration (All): " << duration / 100 << " us" << endl;

	uint64_t mask = 0;

	// libsmctrl_set_global_mask(~0x1ull);
	libsmctrl_set_stream_mask(stream, ~0x1001ull);

	start = chrono::high_resolution_clock::now();

	for (int i = 0; i < 100; i++)
	{
		output = model.forward({ inputs }).toTensor();
		str->synchronize();
	}

	end = chrono::high_resolution_clock::now();
	duration = chrono::duration_cast<chrono::microseconds>(end - start).count();

	cout << "Duration (Masked): " << duration / 100 << " us" << endl;

	return 0;
}

int main2(int argc, char* argv[])
{
	int index = 1;

	type = atoi(argv[index++]);

	// Scheduler::isWretUsed = false;

	if (type == 1)
	{
		taskset = atoi(argv[index++]);
		contextCount = atoi(argv[index++]);
		streamCount = atoi(argv[index++]);
		oversubscription = atof(argv[index++]);
		timer = atoi(argv[index++]);
		windowSize = atoi(argv[index++]);

		warmup = 5;
		repeat = 10;

		if (taskset == 1)
		{
			resnetHighCount = 16;
			resnetLowCount = 32;
			unetHighCount = 0;
			unetLowCount = 0;
			inceptionHighCount = 0;
			inceptionLowCount = 0;
		}

		else if (taskset == 2)
		{
			resnetHighCount = 0;
			resnetLowCount = 0;
			unetHighCount = 5;
			unetLowCount = 10;
			inceptionHighCount = 0;
			inceptionLowCount = 0;
		}

		else if (taskset == 3)
		{
			resnetHighCount = 0;
			resnetLowCount = 0;
			unetHighCount = 0;
			unetLowCount = 0;
			inceptionHighCount = 8;
			inceptionLowCount = 16;
		}

		else if (taskset == 4)
		{
			resnetHighCount = 5;
			resnetLowCount = 11;
			unetHighCount = 2;
			unetLowCount = 3;
			inceptionHighCount = 3;
			inceptionLowCount = 4;
		}

		cout << "Context count: " << contextCount << endl;
		cout << "SM count: " << smCount << endl;
		cout << "Stream count: " << streamCount << endl;
		cout << "Oversubscription: " << oversubscription << endl;

		cout << "Taskset: " << taskset << endl
			<< "\tResHigh: " << resnetHighCount << endl
			<< "\tResLow: " << resnetLowCount << endl
			<< "\tUntHigh: " << unetHighCount << endl
			<< "\tUntLow: " << unetLowCount << endl
			<< "\tIncHigh: " << inceptionHighCount << endl
			<< "\tIncLow: " << inceptionLowCount << endl;
		cout << "-----------------------------" << endl;
	}

	else if (type == 2)
	{
		contextCount = 3;
		streamCount = 1;
		oversubscription = 3;
		maxBatchSize = atoi(argv[index++]);

		warmup = 10;
		repeat = 25;
	}

	else
	{
		cout << "Invalid type." << endl;
		return 0;
	}

	int tempContextCount = max(contextCount, 2);

	smCount = (int)ceil(68 * oversubscription / contextCount);
	smCount += smCount % 2;
	smCount = smCount > 68 ? 68 : smCount;
	cout << "SM count: " << smCount << endl;

	MyContext::mainStreamCount = streamCount;
	// MyContext::secondaryStreamCount = (40 - (contextCount + 1) * streamCount) / contextCount;
	ModuleTracker::windowSize = windowSize;

	int maxStreams[] = { 16, 20, 24, 28, 25, 24, 14, 8 };
	MyContext::secondaryStreamCount = (maxStreams[contextCount - 1] - contextCount * streamCount) / contextCount;

	auto result = Scheduler::initialize(tempContextCount, smCount);

	if (!result)
	{
		cout << "Failed to initialize scheduler." << endl;
		return 0;
	}

	switch (type)
	{
		case 1:
			scenario = Scenario(
				resnetHighCount, resnetLowCount,
				unetHighCount, unetLowCount,
				inceptionHighCount, inceptionLowCount);

			scenario.initialize();
			scenario.analyze(warmup, repeat, contextCount);
			break;

		case 2:
			scenario = Scenario(maxBatchSize);

			scenario.initialize();
			scenario.analyze(warmup, repeat, maxBatchSize);
			return 0;
	}

	scenario.start(timer);

	auto th = thread([&]()
		{
			scenario.maxMemory = max(scenario.maxMemory, (double)Scheduler::getTotalMemoryGB() - Scheduler::getFreeMemoryGB());
			this_thread::sleep_for(chrono::milliseconds(100));
		});

	scenario.wait();
	th.join();

	scenario.saveRecords(taskset, oversubscription);
}