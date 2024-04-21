# pragma once

# include "cnt.hpp"
# include "schd.hpp"
# include "loop.hpp"
# include "resnet.hpp"
# include "unet.hpp"
# include "inception.hpp"

# include <vector>
# include <cuda_profiler_api.h>
# include <fstream>
# include <sys/stat.h>
# include <thread>

using namespace std;
using namespace torch;
using namespace nn;
using namespace FGPRS;

const int
resnetClassCount = 1000,
unetClassCount = 10,
inceptionClassCount = 100,
resnetFrequency = 30,
unetFrequency = 24,
inceptionFrequency = 24;

const string tasksetNames[] = { "res", "unt", "inc", "mix" };

namespace FGPRS
{
	class SummaryReport
	{
	public:
		int throughput;
		double highMissed, lowMissed, acceptanceRate;

		SummaryReport() : throughput(0), highMissed(0), lowMissed(0), acceptanceRate(0) {}
		SummaryReport(int throughput, double highMissed, double lowMissed, double acceptanceRate)
			: throughput(throughput), highMissed(highMissed), lowMissed(lowMissed), acceptanceRate(acceptanceRate)
		{
		}
	};

	class DetailedReport
	{
	public:
		double wret, et, rt;
		bool missed, accepted;

		DetailedReport() : wret(0), et(0), rt(0), missed(false), accepted(false) {}
		DetailedReport(double wret, double et, double rt, bool missed, bool accepted)
			: wret(wret), et(et), rt(rt), missed(missed), accepted(accepted)
		{

		}
	};

	class Scenario
	{
	private:
		int resnetHighCount, resnetLowCount, unetHighCount, unetLowCount, inceptionHighCount, inceptionLowCount;
		int maxBatchSize;
		int timer;
		string mainDirectory;
		string tasksetDirectory[3];

		enum ScenarioType { Main, Batching };
		ScenarioType type;

	public:
		vector<shared_ptr<MyContainer>> highNetworks, lowNetworks, allNetworks;
		vector<shared_ptr<Loop>> loops;

		double maxMemory = 0;

		Scenario() {}

		Scenario(int resnetHighCount, int resnetLowCount,
			int unetHighCount, int unetLowCount,
			int inceptionHighCount, int inceptionLowCount) : type(Main),
			resnetHighCount(resnetHighCount), resnetLowCount(resnetLowCount),
			unetHighCount(unetHighCount), unetLowCount(unetLowCount),
			inceptionHighCount(inceptionHighCount), inceptionLowCount(inceptionLowCount)
		{
		}

		Scenario(int maxBatchSize) : type(Batching),
			maxBatchSize(maxBatchSize)
		{
		}

		~Scenario()
		{
			destroy();
		}

	private:
		void mainInitialize();
		void batchingInitialize();

		void mainAnalyze(int warmup, int repeat, int realContextCount);
		void batchingAnalyze(int warmup, int repeat, int maxBatchSize);

	public:
		void initialize()
		{
			if (type == Main)
				mainInitialize();

			else if (type == Batching)
				batchingInitialize();
		}

		void destroy()
		{
			for (auto mod : allNetworks)
				mod->reset();

			highNetworks.clear();
			lowNetworks.clear();
			allNetworks.clear();
			loops.clear();
		}

		void analyze(int warmup, int repeat, int option)
		{
			if (type == Main)
				mainAnalyze(warmup, repeat, option);

			else if (type == Batching)
				batchingAnalyze(25, 100, option);
		}

		void start(int timer = 0)
		{
			this->timer = timer;

			for (auto loop : loops)
				loop->start(timer);
		}

		void wait()
		{
			for (auto loop : loops)
				loop->wait();
		}

	private:
		SummaryReport summaryReport(int index);
		tuple<vector<DetailedReport>, vector<DetailedReport>> detailedReport(int index);

		void saveSummary(int index)
		{
			auto report = summaryReport(index);

			string directory = tasksetDirectory[index];
			ifstream checkFile(directory + "summary.csv");

			ofstream file(directory + "summary.csv", ios::app);

			if (checkFile.peek() == ifstream::traits_type::eof())
				file << "Throughput, High Missed, Low Missed, Acceptance Rate, Max Memory" << endl;

			file << report.throughput << ", " << report.highMissed * 100 << ", "
				<< report.lowMissed * 100 << ", " << report.acceptanceRate * 100 << ", "
				<< maxMemory << endl;

			ofstream rtHFile(directory + "rtH.csv");
			ofstream rtLFile(directory + "rtL.csv");

			ofstream etHFile(directory + "etH.csv");
			ofstream etLFile(directory + "etL.csv");

			ofstream wretHFile(directory + "wretH.csv");
			ofstream wretLFile(directory + "wretL.csv");

			ofstream wcetHFile(directory + "wcetH.csv");
			ofstream wcetLFile(directory + "wcetL.csv");

			rtHFile << "High Priority Response Time" << endl;
			rtLFile << "Low Priority Response Time" << endl;

			etHFile << "High Priority Execution Time" << endl;
			etLFile << "Low Priority Execution Time" << endl;

			wretHFile << "High Priority WRET" << endl;
			wretLFile << "Low Priority WRET" << endl;

			wcetHFile << "High Priority WCET" << endl;
			wcetLFile << "Low Priority WCET" << endl;

			int highStart, highEnd, lowStart, lowEnd, highCount, lowCount;

			highStart = index == 0 ? 0 : (index == 1 ? resnetHighCount : resnetHighCount + unetHighCount);
			lowStart = index == 0 ? 0 : (index == 1 ? resnetLowCount : resnetLowCount + unetLowCount);

			highEnd = highStart + (index == 0 ? resnetHighCount : (index == 1 ? unetHighCount : inceptionHighCount));
			lowEnd = lowStart + (index == 0 ? resnetLowCount : (index == 1 ? unetLowCount : inceptionLowCount));

			for (int i = highStart; i < highEnd; i++)
			{
				wcetHFile << highNetworks[i]->wcet << endl;

				for (auto record : highNetworks[i]->tracker.records)
				{
					rtHFile << record->responseTime << (record == highNetworks[i]->tracker.records.back() ? "\n" : ", ");
					etHFile << record->executionTime << (record == highNetworks[i]->tracker.records.back() ? "\n" : ", ");
					wretHFile << record->wret << (record == highNetworks[i]->tracker.records.back() ? "\n" : ", ");
				}
			}

			for (int i = lowStart; i < lowEnd; i++)
			{
				bool hasAny = false;

				for (auto record : lowNetworks[i]->tracker.records)
				{
					if (record->accepted)
					{
						hasAny = true;
						break;
					}
				}

				if (!hasAny)
					continue;

				wcetLFile << lowNetworks[i]->wcet << endl;

				for (auto record : lowNetworks[i]->tracker.records)
				{
					if (!record->accepted)
						continue;

					rtLFile << record->responseTime << (record == lowNetworks[i]->tracker.records.back() ? "\n" : ", ");
					etLFile << record->executionTime << (record == lowNetworks[i]->tracker.records.back() ? "\n" : ", ");
					wretLFile << record->wret << (record == lowNetworks[i]->tracker.records.back() ? "\n" : ", ");
				}
			}
		}

		void saveSummaryReport(int taskset)
		{
			if (taskset < 3)
				saveSummary(taskset);

			else
				for (int i = 0; i < 3; i++)
					saveSummary(i);
		}

	public:
		void saveRecords(int taskset, double oversubscribtion)
		{
			mainDirectory = "results";
			mkdir(mainDirectory.c_str(), 0777);

			mainDirectory += "/main";
			mkdir(mainDirectory.c_str(), 0777);

			mainDirectory += "/" + tasksetNames[taskset - 1];
			mkdir(mainDirectory.c_str(), 0777);

			mainDirectory += "/" + to_string(Scheduler::contextCount);
			mkdir(mainDirectory.c_str(), 0777);

			mainDirectory += "/" + to_string(MyContext::mainStreamCount);
			mkdir(mainDirectory.c_str(), 0777);

			mainDirectory += "/" + to_string(int(oversubscribtion));

			int digit = int(oversubscribtion * 10) % 10;

			if (digit != 0)
			{
				mainDirectory += "." + to_string(digit);
				digit = int(oversubscribtion * 100) % 10;

				if (digit != 0)
					mainDirectory += to_string(digit);
			}

			mainDirectory += "/";
			mkdir(mainDirectory.c_str(), 0777);

			for (int i = 0; i < 3; i++)
			{
				tasksetDirectory[i] = mainDirectory + (taskset < 4 ? "" : tasksetNames[i] + "/");
				mkdir(tasksetDirectory[i].c_str(), 0777);
			}

			saveSummaryReport(taskset - 1);
		}
	};
}