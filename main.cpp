# include <iostream>
# include <vector>
# include <cuda_profiler_api.h>

# include "cnt.hpp"
# include "schd.hpp"
# include "resnet.hpp"
# include "loop.hpp"

using namespace std;
using namespace FGPRS;

int contextCount, smCount, moduleCount, highCount, lowCount, streamCount;
double oversubscription, highPercentage;
vector<shared_ptr<MyContainer>> highNetworks, lowNetworks, networks;
vector<shared_ptr<Loop>> loops;
int timer, windowSize;

const int warmup = 5, repeat = 10;

int main(int argc, char* argv[])
{
	// cudaProfilerStart();
	contextCount = atoi(argv[1]);
	oversubscription = atof(argv[2]);
	moduleCount = atoi(argv[3]);
	highPercentage = atof(argv[4]) / 100.0;
	streamCount = atoi(argv[5]);
	timer = atoi(argv[6]);
	windowSize = atoi(argv[7]);

	// moduleCount = contextCount * streamCount;

	smCount = (int)ceil(68 * oversubscription / contextCount);
	smCount += smCount % 2;
	smCount = smCount > 68 ? 68 : smCount;

	highCount = (int)ceil(moduleCount * highPercentage);
	lowCount = moduleCount - highCount;
	MyContext::streamCount = streamCount;
	ModuleTracker::windowSize = windowSize;

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

	for (int i = 0; i < highCount; i++)
	{
		highNetworks.push_back(resnet18(1000));
		highNetworks[i]->initialize(highNetworks[i], "resH" + to_string(i + 1), true);
		highNetworks[i]->setFrequency(30);
		highNetworks[i]->inputSize = 224;
		networks.push_back(highNetworks[i]);
	}

	for (int i = 0; i < lowCount; i++)
	{
		lowNetworks.push_back(resnet18(1000));
		lowNetworks[i]->initialize(lowNetworks[i], "resL" + to_string(i + 1), false);
		lowNetworks[i]->setFrequency(30);
		lowNetworks[i]->inputSize = 224;
		networks.push_back(lowNetworks[i]);
	}

	cout << "Scheduler initialized." << endl;
	Scheduler::populateModules(highNetworks, lowNetworks);

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

	for (auto mod : networks)
		mod->assignDeadline();

	for (auto mod : networks)
		loops.push_back(make_shared<Loop>(Loop(mod)));

	// cudaProfilerStart();

	for (auto loop : loops)
		loop->start(timer);

	for (auto loop : loops)
		loop->wait();

	// cudaProfilerStop();

	cout << endl << endl << endl << "-----------------------------" << endl;
	cout << "High priority modules:" << endl << endl;

	for (auto mod : highNetworks)
	{
		int minExec = INT_MAX, maxExec = 0, aveExec = 0;
		int minResp = INT_MAX, maxResp = 0, aveResp = 0;

		for (auto rec : mod->tracker.records)
		{
			minExec = rec->executionTime < minExec ? rec->executionTime : minExec;
			maxExec = rec->executionTime > maxExec ? rec->executionTime : maxExec;
			aveExec += rec->executionTime;

			minResp = rec->responseTime < minResp ? rec->responseTime : minResp;
			maxResp = rec->responseTime > maxResp ? rec->responseTime : maxResp;
			aveResp += rec->responseTime;
		}

		aveExec /= mod->tracker.records.size();
		aveResp /= mod->tracker.records.size();
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

		for (auto rec : mod->tracker.records)
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

		for (auto rec : mod->tracker.records)
		{
			if (!rec->accepted)
				skipped++;

			if (rec->missed)
				missed++;
		}

		minAcceptance = mod->acceptanceRate < minAcceptance ? mod->acceptanceRate : minAcceptance;
		maxAcceptance = mod->acceptanceRate > maxAcceptance ? mod->acceptanceRate : maxAcceptance;
		aveAcceptance += mod->acceptanceRate;

		cout << "Module \"" << mod->moduleName << "\":" << endl
			<< "\tSkipped: " << skipped << " times" << endl
			<< "\tMissed: " << missed << " times" << endl
			<< "\tAccepted Rated: " << mod->acceptanceRate << endl;
	}

	aveAcceptance /= lowNetworks.size();
	cout << "Minimum acceptance rate: " << minAcceptance << endl
		<< "Maximum acceptance rate: " << maxAcceptance << endl
		<< "Average acceptance rate: " << aveAcceptance << endl;
}