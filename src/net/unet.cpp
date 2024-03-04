# include "unet.hpp"

using namespace FGPRS;

void UNet::initialize(shared_ptr<MyContainer> module, string name, bool highPriority)
{
	moduleName = name;
	this->highPriority = highPriority;

	for (int i = 0; i < (levels - 1); i++)
	{
		oContracting.push_back(make_shared<Operation>("contracting" + to_string(i), module, contracting[i].ptr(), false));
		oDownsampling.push_back(make_shared<Operation>("downsampling" + to_string(i), module, downsampling[i].ptr(), false));

		addOperation(oContracting.back());
		addOperation(oDownsampling.back());
	}

	oBottleneck = make_shared<Operation>("bottleneck", module, bottleneck.ptr(), false);
	addOperation(oBottleneck);

	for (int i = 0; i < (levels - 1); i++)
	{
		oUpsampling.push_back(make_shared<Operation>("upsampling" + to_string(i), module, upsampling[i].ptr(), false));
		oExpanding.push_back(make_shared<Operation>("expanding" + to_string(i), module, expanding[i].ptr(), false));

		addOperation(oUpsampling.back());
		addOperation(oExpanding.back());
	}

	oOutput = make_shared<Operation>("output", module, seqOutput.ptr(), true);
	addOperation(oOutput);
}

void UNet::analyzeOperations(int warmup, int repeat, bool isWcet)
{
	Tensor inputTensor = torch::randn({ 1, 3, inputSize, inputSize }, kCUDA);
	int timer;

	vector<Tensor> contractingTensor(levels - 1);
	vector<Tensor> downsamplingTensor(levels - 1);
	Tensor bottleneckTensor;
	vector<Tensor> upsamplingTensor(levels - 1);
	vector<Tensor> expandingTensor(levels - 1);

	isWcet ? wcet = 0 : bcet = 0;

	for (int level = 0; level < levels - 1; level++)
	{
		contractingTensor[level] =
			oContracting[level]->analyze(
				warmup, repeat, level == 0 ? inputTensor : downsamplingTensor[level - 1], &timer);
		isWcet ? oContracting[level]->wcet = timer : oContracting[level]->bcet = timer;
		isWcet ? wcet += timer : bcet += timer;

		downsamplingTensor[level] = oDownsampling[level]->analyze(warmup, repeat, contractingTensor[level], &timer);
		isWcet ? oDownsampling[level]->wcet = timer : oDownsampling[level]->bcet = timer;
		isWcet ? wcet += timer : bcet += timer;
	}

	bottleneckTensor = oBottleneck->analyze(warmup, repeat, downsamplingTensor.back(), &timer);
	isWcet ? oBottleneck->wcet = timer : oBottleneck->bcet = timer;
	isWcet ? wcet += timer : bcet += timer;

	for (int level = levels - 2; level >= 0; level--)
	{
		upsamplingTensor[level] = oUpsampling[level]->analyze(
			warmup, repeat, level == levels - 2 ? bottleneckTensor : expandingTensor[level + 1], &timer);
		isWcet ? oUpsampling[level]->wcet = timer : oUpsampling[level]->bcet = timer;
		isWcet ? wcet += timer : bcet += timer;

		if (paddingSize == 0)
		{
			int oldXSize = contractingTensor[level].size(2);
			int oldYSize = contractingTensor[level].size(3);
			int newXSize = upsamplingTensor[level].size(2);
			int newYSize = upsamplingTensor[level].size(3);
			int startX = oldXSize / 2 - newXSize / 2;
			int startY = oldYSize / 2 - newYSize / 2;
			contractingTensor[level] = contractingTensor[level].slice(2, startX, startX + newXSize);
			contractingTensor[level] = contractingTensor[level].slice(3, startY, startY + newYSize);
		}

		expandingTensor[level] = oExpanding[level]->analyze(
			warmup, repeat, cat({ contractingTensor[level],upsamplingTensor[level] }, 1), &timer);
		isWcet ? oExpanding[level]->wcet = timer : oExpanding[level]->bcet = timer;
		isWcet ? wcet += timer : bcet += timer;
	}

	oOutput->analyze(warmup, repeat, expandingTensor.front(), &timer);
	isWcet ? oOutput->wcet = timer : oOutput->bcet = timer;
	isWcet ? wcet += timer : bcet += timer;
}

Tensor UNet::forwardDummy(Tensor inputTensor, MyStream* str)
{
	vector<Tensor> contractingTensor(levels - 1);
	vector<Tensor> downsamplingTensor(levels - 1);
	Tensor bottleneckTensor;
	vector<Tensor> upsamplingTensor(levels - 1);
	vector<Tensor> expandingTensor(levels - 1);
	Tensor outputTensor;

	for (int level = 0; level < levels - 1; level++)
	{
		contractingTensor[level] = contracting[level]->forward(level == 0 ? inputTensor : downsamplingTensor[level - 1]);
		str->synchronize();
		downsamplingTensor[level] = downsampling[level]->forward(contractingTensor[level]);
		str->synchronize();
	}

	bottleneckTensor = bottleneck->forward(downsamplingTensor.back());
	str->synchronize();

	for (int level = levels - 2; level >= 0; level--)
	{
		upsamplingTensor[level] = upsampling[level]->forward(level == levels - 2 ? bottleneckTensor : expandingTensor[level + 1]);
		str->synchronize();

		if (paddingSize == 0)
		{
			int oldXSize = contractingTensor[level].size(2);
			int oldYSize = contractingTensor[level].size(3);
			int newXSize = upsamplingTensor[level].size(2);
			int newYSize = upsamplingTensor[level].size(3);
			int startX = oldXSize / 2 - newXSize / 2;
			int startY = oldYSize / 2 - newYSize / 2;
			contractingTensor[level] = contractingTensor[level].slice(2, startX, startX + newXSize);
			str->synchronize();
			contractingTensor[level] = contractingTensor[level].slice(3, startY, startY + newYSize);
			str->synchronize();
		}

		expandingTensor[level] = expanding[level]->forward(cat({ contractingTensor[level],upsamplingTensor[level] }, 1));
		str->synchronize();
	}

	outputTensor = output->forward(expandingTensor.front());
	str->synchronize();

	return outputTensor;
}

Tensor UNet::releaseOperations(Tensor inputTensor)
{
	auto opPrev = operations.back();

	vector<Tensor> contractingTensor(levels - 1);
	vector<Tensor> downsamplingTensor(levels - 1);
	Tensor bottleneckTensor;
	vector<Tensor> upsamplingTensor(levels - 1);
	vector<Tensor> expandingTensor(levels - 1);
	Tensor outputTensor;

	for (int level = 0; level < levels - 1; level++)
	{
		oContracting[level]->priorDelayed = opPrev->delayed;
		contractingTensor[level] =
			oContracting[level]->releaseSync(level == 0 ? inputTensor : downsamplingTensor[level - 1]);

		oDownsampling[level]->priorDelayed = oContracting[level]->delayed;
		downsamplingTensor[level] = oDownsampling[level]->releaseSync(contractingTensor[level]);

		opPrev = oDownsampling[level];
	}

	oBottleneck->priorDelayed = oDownsampling.back()->delayed;
	bottleneckTensor = oBottleneck->releaseSync(downsamplingTensor.back());

	for (int level = levels - 2; level >= 0; level--)
	{
		oUpsampling[level]->priorDelayed = opPrev->delayed;
		upsamplingTensor[level] = oUpsampling[level]->releaseSync(level == levels - 2 ? bottleneckTensor : expandingTensor[level + 1]);

		if (paddingSize == 0)
		{
			int oldXSize = contractingTensor[level].size(2);
			int oldYSize = contractingTensor[level].size(3);
			int newXSize = upsamplingTensor[level].size(2);
			int newYSize = upsamplingTensor[level].size(3);
			int startX = oldXSize / 2 - newXSize / 2;
			int startY = oldYSize / 2 - newYSize / 2;
			contractingTensor[level] = contractingTensor[level].slice(2, startX, startX + newXSize);
			contractingTensor[level] = contractingTensor[level].slice(3, startY, startY + newYSize);
		}

		oExpanding[level]->priorDelayed = oUpsampling[level]->delayed;
		expandingTensor[level] = oExpanding[level]->releaseSync(cat({ contractingTensor[level],upsamplingTensor[level] }, 1));

		opPrev = oExpanding[level];
	}

	oOutput->priorDelayed = opPrev->delayed;
	outputTensor = oOutput->releaseSync(expandingTensor.front());

	return outputTensor;
}

shared_ptr<UNet> unet(int numClasses)
{
	auto model = make_shared<UNet>(UNet(3, numClasses));
	return model;
}