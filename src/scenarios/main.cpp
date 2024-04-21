# include "scenario.hpp"

using namespace FGPRS;

void Scenario::mainInitialize()
{
	auto ctx = Scheduler::selectDefaultContext();
	ctx->select();

	for (int i = 0; i < resnetHighCount; i++)
	{
		highNetworks.push_back(resnet18(resnetClassCount));
		highNetworks.back()->initialize(highNetworks.back(), "HighRes" + to_string(i + 1), true);
		highNetworks.back()->setFrequency(resnetFrequency);
		highNetworks.back()->inputSize = 224;
		highNetworks.back()->batchSize = 1;
	}

	for (int i = 0; i < unetHighCount; i++)
	{
		highNetworks.push_back(unet(unetClassCount));
		highNetworks.back()->initialize(highNetworks.back(), "HighUnt" + to_string(i + 1), true);
		highNetworks.back()->setFrequency(unetFrequency);
		highNetworks.back()->inputSize = 224;
		highNetworks.back()->batchSize = 1;
	}

	for (int i = 0; i < inceptionHighCount; i++)
	{
		highNetworks.push_back(inception3(inceptionClassCount));
		highNetworks.back()->initialize(highNetworks.back(), "HighInc" + to_string(i + 1), true);
		highNetworks.back()->setFrequency(inceptionFrequency);
		highNetworks.back()->inputSize = 224;
		highNetworks.back()->batchSize = 1;
	}

	for (int i = 0; i < resnetLowCount; i++)
	{
		lowNetworks.push_back(resnet18(resnetClassCount));
		lowNetworks.back()->initialize(lowNetworks.back(), "LowRes" + to_string(i + 1), false);
		lowNetworks.back()->setFrequency(resnetFrequency);
		lowNetworks.back()->inputSize = 224;
		lowNetworks.back()->batchSize = 1;
	}

	for (int i = 0; i < unetLowCount; i++)
	{
		lowNetworks.push_back(unet(unetClassCount));
		lowNetworks.back()->initialize(lowNetworks.back(), "LowUnt" + to_string(i + 1), false);
		lowNetworks.back()->setFrequency(unetFrequency);
		lowNetworks.back()->inputSize = 224;
		lowNetworks.back()->batchSize = 1;
	}

	for (int i = 0; i < inceptionLowCount; i++)
	{
		lowNetworks.push_back(inception3(inceptionClassCount));
		lowNetworks.back()->initialize(lowNetworks.back(), "LowInc" + to_string(i + 1), false);
		lowNetworks.back()->setFrequency(inceptionFrequency);
		lowNetworks.back()->inputSize = 224;
		lowNetworks.back()->batchSize = 1;
	}

	allNetworks.insert(allNetworks.end(), highNetworks.begin(), highNetworks.end());
	allNetworks.insert(allNetworks.end(), lowNetworks.begin(), lowNetworks.end());
}

void Scenario::mainAnalyze(int warmup, int repeat, int realContextCount)
{
	Scheduler::populateModulesByOrder(highNetworks, lowNetworks);

	for (auto mod : allNetworks)
		mod->analyzeBCET(1, 1);

	for (auto mod : allNetworks)
		mod->analyzeWCET(1, 1);

	if (realContextCount == 1)
	{
		Scheduler::contextCount = 1;
		Scheduler::populateModulesByUtilization(highNetworks, lowNetworks);
	}

	for (auto mod : allNetworks)
	{
		mod->analyzeBCET(warmup, repeat);
		// cout << mod->moduleName << " BCET: " << mod->bcet << " us" << endl;
	}

	for (auto mod : allNetworks)
	{
		mod->analyzeWCET(warmup, repeat);
		// cout << mod->moduleName << " WCET: " << mod->wcet << " us" << endl;
	}

	Scheduler::populateModulesByUtilization(highNetworks, lowNetworks);

	for (auto mod : allNetworks)
		mod->assignDeadline();

	if (loops.size() == 0)
		for (auto mod : allNetworks)
			loops.push_back(make_shared<Loop>(mod));
}

SummaryReport Scenario::summaryReport(int index)
{
	int throughput = 0;
	double highMissed, lowMissed, acceptanceRate;
	int highStart, highEnd, lowStart, lowEnd, highCount, lowCount;

	highCount = index == 0 ? resnetHighCount :
		(index == 1 ? unetHighCount : inceptionHighCount);
	lowCount = index == 0 ? resnetLowCount :
		(index == 1 ? unetLowCount : inceptionLowCount);

	highStart = index == 0 ? 0 : (index == 1 ? resnetHighCount : resnetHighCount + unetHighCount);
	highEnd = highStart + highCount;

	lowStart = index == 0 ? 0 : (index == 1 ? resnetLowCount : resnetLowCount + unetLowCount);
	lowEnd = lowStart + lowCount;

	int accepted, missed = 0;

	for (int i = highStart; i < highEnd; i++)
	{
		throughput += highNetworks[i]->tracker.records.size();

		for (auto rec : highNetworks[i]->tracker.records)
			missed += rec->missed;
	}

	highMissed = (double)missed / (double)throughput;

	int highThroughput = throughput, lowArrived = 0;
	missed = 0;

	for (int i = lowStart; i < lowEnd; i++)
	{
		lowArrived += lowNetworks[i]->tracker.records.size();

		for (auto rec : lowNetworks[i]->tracker.records)
		{
			throughput += rec->accepted;
			missed += rec->missed;
		}
	}

	lowMissed = (double)missed / (double)(throughput - highThroughput);
	acceptanceRate = (double)(throughput - highThroughput) / (double)lowArrived;
	throughput /= (timer / 1e3);

	cout << "Throughput: " << throughput << " tasks/s" << endl;
	cout << "High Missed: " << highMissed << endl;
	cout << "Low Missed: " << lowMissed << endl;
	cout << "Acceptance Rate: " << acceptanceRate << endl;

	return SummaryReport{ throughput, highMissed, lowMissed, acceptanceRate };
}

tuple<vector<DetailedReport>, vector<DetailedReport>> Scenario::detailedReport(int index)
{
	vector<DetailedReport> outHigh, outLow;
	int highStart, highEnd, lowStart, lowEnd;

	highStart = index == 0 ? 0 : (index == 1 ? resnetHighCount : resnetHighCount + unetHighCount);
	highEnd = highStart + (index == 0 ? resnetHighCount : (index == 1 ? unetHighCount : inceptionHighCount));

	lowStart = index == 0 ? 0 : (index == 1 ? resnetLowCount : resnetLowCount + unetLowCount);
	lowEnd = lowStart + (index == 0 ? resnetLowCount : (index == 1 ? unetLowCount : inceptionLowCount));

	for (int i = highStart; i < highEnd; i++)
		for (auto rec : highNetworks[i]->tracker.records)
			outHigh.push_back(DetailedReport{ rec->wret / 1000., rec->executionTime / 1000.,
				rec->responseTime / 1000., rec->missed, rec->accepted });

	for (int i = lowStart; i < lowEnd; i++)
		for (auto rec : lowNetworks[i]->tracker.records)
			outLow.push_back(DetailedReport{ rec->wret / 1000., rec->executionTime / 1000.,
				rec->responseTime / 1000., rec->missed, rec->accepted });

	return make_tuple(outHigh, outLow);
}