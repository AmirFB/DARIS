# include "scenario.hpp"

using namespace FGPRS;

void Scenario::batchingInitialize()
{
	auto ctx = Scheduler::selectDefaultContext();
	ctx->select();

	// allNetworks.push_back(resnet18(resnetClassCount));
	// allNetworks.push_back(unet(unetClassCount));
	allNetworks.push_back(inception3(inceptionClassCount));
	// allNetworks.push_back(resnet50(resnetClassCount));

	for (int i = 0; i < allNetworks.size(); i++)
	{
		allNetworks[i]->initialize(allNetworks[i], "Batch" + to_string(i + 1), true, 1);
		allNetworks[i]->inputSize = 448;
		allNetworks[i]->batchSize = 1;
	}
}

void Scenario::batchingAnalyze(int warmup, int repeat, int maxBatchSize)
{
	vector<vector<double>> output;

	Scheduler::populateModulesByOrder(allNetworks, lowNetworks);

	for (int i = 0; i < allNetworks.size(); i++)
		allNetworks[i]->analyzeBCET(1, 10);

	for (int i = 0; i < allNetworks.size(); i++)
	{
		output.push_back(vector<double>());

		cout << (i == 0 ? "ResNet" : (i == 1 ? "UNet" : (i == 2 ? "Inception" : "ResNet50"))) << endl;

		for (int batch = 1; batch <= maxBatchSize; batch++)
		{
			allNetworks[i]->batchSize = batch;
			allNetworks[i]->analyzeBCET(warmup, repeat);

			cout << "\t" << batch << "\tFPS: " << (1000000. / allNetworks[i]->bcet * batch) << " \tT=" << allNetworks[i]->bcet << "us" << endl;

			output[i].push_back(1000000. / allNetworks[i]->bcet * batch);
		}

		cout << "Gain: " << (double)output[i].back() / output[i].front() << endl << endl;
	}

	mkdir("results", 0777);
	mkdir("results/batching", 0777);

	ifstream checkFile("results/batching/inc448.csv");

	// ofstream resFile("results/batching/res.csv", ios::app);
	// ofstream untFile("results/batching/unt.csv", ios::app);
	ofstream incFile("results/batching/inc448.csv", ios::app);
	// ofstream res50File("results/batching/res50.csv", ios::app);

	if (!checkFile.peek() == ifstream::traits_type::eof())
	{
		// resFile << "Batch, FPS" << endl;
		// untFile << "Batch, FPS" << endl;
		incFile << "Batch, FPS" << endl;
		// res50File << "Batch, FPS" << endl;
	}

	for (int i = 0; i < maxBatchSize; i++)
	{
		// resFile << output[0][i] << (i == maxBatchSize - 1 ? "" : ", ");
		// untFile << output[1][i] << (i == maxBatchSize - 1 ? "" : ", ");
		incFile << output[0][i] << (i == maxBatchSize - 1 ? "" : ", ");
		// res50File << output[3][i] << (i == maxBatchSize - 1 ? "" : ", ");
	}

	// resFile << endl;
	// untFile << endl;
	incFile << endl;
	// res50File << endl;
}