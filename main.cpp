# include <iostream>
# include <vector>
# include <cuda_profiler_api.h>
# include <torch/script.h>

# include "cnt.hpp"
# include "schd.hpp"
# include "resnet.hpp"
# include "unet.hpp"
# include "inception.hpp"
# include "loop.hpp"
# include "can.hpp"

using namespace std;
using namespace FGPRS;

int contextCount, smCount, moduleCount, highCount, lowCount, streamCount;
int resnetCount, resnetLowCount, resnetHighCount, unetCount, unetLowCount, unetHighCount;
double oversubscription, highPercentage;
vector<shared_ptr<MyContainer>> highNetworks, lowNetworks, networks;
vector<shared_ptr<Loop>> loops;
int timer, windowSize;

const int warmup = 5, repeat = 10;
const int frequency = 24, inputSize = 224;

vector<string> types = { "resnet", "unet" };

int main(int argc, char* argv[])
{
	int index = 1;
	int scenario = atoi(argv[index++]);

	contextCount = atoi(argv[index++]);
	oversubscription = atof(argv[index++]);

	if (scenario == 1)
	{
		resnetCount = atoi(argv[index++]);
		unetCount = atoi(argv[index++]);
		highPercentage = atof(argv[index++]) / 100.0;

		moduleCount = resnetCount + unetCount;
		resnetHighCount = (int)ceil(resnetCount * highPercentage);
		resnetLowCount = resnetCount - resnetHighCount;
		unetHighCount = (int)ceil(unetCount * highPercentage);
		unetLowCount = unetCount - unetHighCount;
		highCount = resnetHighCount + unetHighCount;
		lowCount = resnetLowCount + unetLowCount;
	}

	streamCount = atoi(argv[index++]);
	timer = atoi(argv[index++]);
	windowSize = atoi(argv[index++]);

	smCount = (int)ceil(68 * oversubscription / contextCount);
	smCount += smCount % 2;
	smCount = smCount > 68 ? 68 : smCount;

	highCount = (int)ceil(moduleCount * highPercentage);
	lowCount = moduleCount - highCount;
	MyContext::streamCount = streamCount;
	ModuleTracker::windowSize = windowSize;

	// auto res = resnet18(1000);
	// auto un = unet(1000);
	// auto input = torch::randn({ 1, 3, 224, 224 }, kCUDA);

	// res->eval();
	// un->eval();
	// res->to(at::kCUDA);
	// un->to(at::kCUDA);

	// for (int i = 0; i < 100; i++)
	// {
	// 	auto output = res->forward(input);
	// 	cudaDeviceSynchronize();
	// 	output = un->forward(input);
	// 	cudaDeviceSynchronize();
	// }

	// auto start = chrono::high_resolution_clock::now();

	// for (int i = 0; i < 1000; i++)
	// {
	// 	auto output = res->forward(input);
	// 	cudaDeviceSynchronize();
	// }

	// auto end = chrono::high_resolution_clock::now();
	// auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
	// auto fps = 1000000.0 / duration.count() * 1000;

	// cout << "Resnet18:" << endl
	// 	<< "\t" << fps << " fps" << endl
	// 	<< "\t" << duration.count() / 1000 << " us" << endl;

	// start = chrono::high_resolution_clock::now();

	// for (int i = 0; i < 1000; i++)
	// {
	// 	auto output = un->forward(input);
	// 	cudaDeviceSynchronize();
	// }

	// end = chrono::high_resolution_clock::now();
	// duration = chrono::duration_cast<chrono::microseconds>(end - start);
	// fps = 1000000.0 / duration.count() * 1000;

	// cout << "Unet:" << endl
	// 	<< "\t" << fps << " fps" << endl
	// 	<< "\t" << duration.count() / 1000 << " us" << endl;

	// return 0;

	cout << "Context count: " << contextCount << endl;
	cout << "SM count: " << smCount << endl;
	cout << "Module count: " << moduleCount << endl;
	cout << "High priority module count: " << highCount << endl;
	cout << "Low priority module count: " << lowCount << endl;
	cout << "Stream count: " << streamCount << endl;
	cout << "-----------------------------" << endl;

	cuInit(0);

	auto result = Scheduler::initialize(contextCount, smCount);

	if (!result)
	{
		cout << "Failed to initialize scheduler." << endl;
		return 0;
	}

	auto ctx = Scheduler::selectDefaultContext();
	ctx->select();

	for (int i = 0; i < resnetHighCount; i++)
	{
		highNetworks.push_back(resnet18(1000));
		highNetworks.back()->initialize(highNetworks.back(), "Hresnet" + to_string(i + 1), true);
		highNetworks.back()->setFrequency(frequency);
		highNetworks.back()->inputSize = inputSize;
		networks.push_back(highNetworks.back());
	}

	for (int i = 0; i < unetHighCount; i++)
	{
		highNetworks.push_back(unet(1000));
		highNetworks.back()->initialize(highNetworks.back(), "Hunet" + to_string(i + 1), true);
		highNetworks.back()->setFrequency(frequency);
		highNetworks.back()->inputSize = inputSize;
		networks.push_back(highNetworks.back());
	}

	for (int i = 0; i < resnetLowCount; i++)
	{
		lowNetworks.push_back(resnet18(1000));
		lowNetworks.back()->initialize(lowNetworks.back(), "Lresnet" + to_string(i + 1), false);
		lowNetworks.back()->setFrequency(frequency);
		lowNetworks.back()->inputSize = inputSize;
		networks.push_back(lowNetworks.back());
	}

	for (int i = 0; i < unetLowCount; i++)
	{
		lowNetworks.push_back(unet(1000));
		lowNetworks.back()->initialize(lowNetworks.back(), "Lunet" + to_string(i + 1), false);
		lowNetworks.back()->setFrequency(frequency);
		lowNetworks.back()->inputSize = inputSize;
		networks.push_back(lowNetworks.back());
	}

	cout << "Scheduler initialized." << endl;
	Scheduler::populateModulesByOrder(highNetworks, lowNetworks);

	for (auto mod : networks)
		mod->analyzeBCET(1, 1);

	for (auto mod : networks)
	{
		mod->analyzeBCET(warmup, repeat);
		cout << "\"" << mod->moduleName << "\" BCET: " << mod->bcet << " us" << endl;
	}

	for (auto mod : networks)
		mod->analyzeWCET(1, 1);

	for (auto mod : networks)
	{
		mod->analyzeWCET(warmup, repeat);
		cout << "\"" << mod->moduleName << "\" WCET: " << mod->wcet << " us" << endl;
	}

	Scheduler::populateModulesByUtilization(highNetworks, lowNetworks);

	for (auto mod : networks)
		mod->assignDeadline();

	for (auto mod : networks)
		loops.push_back(make_shared<Loop>(Loop(mod)));

	for (auto loop : loops)
		loop->start(timer);

	for (auto loop : loops)
		loop->wait();

	cout << endl << endl << endl << "-----------------------------" << endl;
	cout << "High priority modules:" << endl << endl;

	for (auto mod : highNetworks)
	{
		int minExec = INT_MAX, maxExec = 0, aveExec = 0;
		int minResp = INT_MAX, maxResp = 0, aveResp = 0;

		for (auto rec : mod->tracker.finalRecords)
		{
			minExec = rec->executionTime < minExec ? rec->executionTime : minExec;
			maxExec = rec->executionTime > maxExec ? rec->executionTime : maxExec;
			aveExec += rec->executionTime;

			minResp = rec->responseTime < minResp ? rec->responseTime : minResp;
			maxResp = rec->responseTime > maxResp ? rec->responseTime : maxResp;
			aveResp += rec->responseTime;
		}

		aveExec /= mod->tracker.finalRecords.size();
		aveResp /= mod->tracker.finalRecords.size();
		cout << "Module \"" << mod->moduleName << "\":" << endl
			<< "\tMinimum execution time: " << minExec << " us" << endl
			<< "\tMaximum execution time: " << maxExec << " us" << endl
			<< "\tAverage execution time: " << aveExec << " us" << endl
			<< "\tWorst Recent Execution: " << mod->wret << " us" << endl
			<< "\tMinimum response time: " << minResp << " us" << endl
			<< "\tMaximum response time: " << maxResp << " us" << endl
			<< "\tAverage response time: " << aveResp << " us" << endl << endl;
	}

	cout << endl << endl << endl << "-----------------------------" << endl;
	cout << "Low priority modules:" << endl << endl;

	for (auto mod : lowNetworks)
	{
		int minExec = INT_MAX, maxExec = 0, aveExec = 0;
		int minResp = INT_MAX, maxResp = 0, aveResp = 0;
		int count = 0;

		for (auto rec : mod->tracker.finalRecords)
		{
			if (!rec->accepted)
				continue;

			count++;
			minExec = rec->executionTime < minExec ? rec->executionTime : minExec;
			maxExec = rec->executionTime > maxExec ? rec->executionTime : maxExec;
			aveExec += rec->executionTime;

			minResp = rec->responseTime < minResp ? rec->responseTime : minResp;
			maxResp = rec->responseTime > maxResp ? rec->responseTime : maxResp;
			aveResp += rec->responseTime;
		}

		if (count == 0)
			count = 1;

		aveExec /= count;
		aveResp /= count;
		cout << "Module \"" << mod->moduleName << "\":" << endl
			<< "\tMinimum execution time: " << minExec << " us" << endl
			<< "\tMaximum execution time: " << maxExec << " us" << endl
			<< "\tAverage execution time: " << aveExec << " us" << endl
			<< "\tWorst Recent Execution: " << mod->wret << " us" << endl
			<< "\tMinimum response time: " << minResp << " us" << endl
			<< "\tMaximum response time: " << maxResp << " us" << endl
			<< "\tAverage response time: " << aveResp << " us" << endl << endl;
	}

	cout << endl << endl << endl << "-----------------------------" << endl;

	double minAcceptance = 10, maxAcceptance = 0, aveAcceptance = 0;

	for (auto mod : lowNetworks)
	{
		int skipped = 0, missed = 0;

		for (auto rec : mod->tracker.finalRecords)
		{
			if (!rec->accepted)
				skipped++;

			if (rec->missed)
				missed++;
		}

		minAcceptance = mod->acceptanceRate < minAcceptance ? mod->acceptanceRate : minAcceptance;
		maxAcceptance = mod->acceptanceRate > maxAcceptance ? mod->acceptanceRate : maxAcceptance;
		aveAcceptance += mod->acceptanceRate;

		// cout << "Module \"" << mod->moduleName << "\":" << endl
		// 	<< "\tSkipped: " << skipped << " times" << endl
		// 	<< "\tMissed: " << missed << " times" << endl
		// 	<< "\tAccepted Rated: " << mod->acceptanceRate << endl;
	}

	aveAcceptance /= lowNetworks.size();
	cout << "Minimum acceptance rate: " << minAcceptance << endl
		<< "Maximum acceptance rate: " << maxAcceptance << endl
		<< "Average acceptance rate: " << aveAcceptance << endl;

	for (int i = 0; i < Scheduler::contextCount; i++)
	{
		auto ctx = &Scheduler::contextPool[i];

		cout << "Context " << ctx->index << ":" << endl
			<< "\tUtilization: " << ctx->overallUtilization << endl
			<< "\tAcceptance Rate: " << ctx->acceptanceRate << endl;
		// << "\tMissed: " << ctx->missedCount << endl
		// << "\tAccepted: " << ctx->acceptedCount << endl;

		for (auto mod : ctx->highContainers)
		{
			cout << "\t\t\"" << mod->moduleName << "\"" << endl
				<< "\t\t\tUtilization: " << mod->utilizationPartitioned << endl;
		}

		for (auto mod : ctx->lowContainers)
		{
			cout << "\t\t\"" << mod->moduleName << "\":" << endl
				<< "\t\t\tUtilization: " << mod->utilizationPartitioned << endl
				<< "\t\t\tAcceptance Rate: " << mod->acceptanceRate << endl;
			// << "\t\t\tMissed: " << mod->missedCount << endl
			// << "\t\t\tSkipped: " << mod->skippedCount << endl;
		}
	}
}