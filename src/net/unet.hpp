# pragma once

# include <torch/torch.h>

# include "unetConv.hpp"

# include "cnt.hpp"

using namespace std;
using namespace torch;
using namespace nn;
using namespace FGPRS;

struct UNet : MyContainer
{
private:
	int levels;
	int kernelSize;
	int paddingSize;

	bool convolutionDownsampling;
	bool convolutionUpsampling;
	bool partialConvolution;
	bool batchNorm;
	bool showSizes;

	deque<Sequential> contracting;
	deque<Sequential> downsampling;
	Sequential bottleneck;
	deque<Sequential> upsampling;
	deque<Sequential> expanding;
	Conv2d output{ nullptr };
	Sequential seqOutput{ nullptr };

public:
	vector<shared_ptr<Operation>> oContracting, oDownsampling, oUpsampling, oExpanding;
	shared_ptr<Operation> oBottleneck, oOutput;

	UNet(int32_t inChannels, int32_t outChannels, int32_t featureChannels = 32,
		int32_t levels = 4, int32_t kernelSize = 3, bool padding = true,
		bool convolutionDownsampling = false, bool convolutionUpsampling = false,
		bool partialConvolution = true, bool batchNorm = false, bool showSizes = false)
	{
		this->levels = levels;
		this->kernelSize = kernelSize;
		this->paddingSize = padding ? (kernelSize - 1) / 2 : 0;

		this->convolutionDownsampling = convolutionDownsampling;
		this->convolutionUpsampling = convolutionUpsampling;
		this->partialConvolution = partialConvolution;
		this->batchNorm = batchNorm;
		this->showSizes = showSizes;

		for (int level = 0; level < levels - 1; level++)
		{
			contracting.push_back(levelBlock(level == 0 ? inChannels : featureChannels * (1 << (level - 1)), featureChannels * (1 << level)));
			register_module("contractingBlock" + to_string(level), contracting.back());

			downsampling.push_back(downsamplingBlock(featureChannels * (1 << level)));
			register_module("downsampling" + to_string(level), downsampling.back());
		}

		bottleneck = levelBlock(featureChannels * (1 << (levels - 2)), featureChannels * (1 << (levels - 1)));
		register_module("bottleneck", bottleneck);

		for (int level = levels - 2; level >= 0; level--)
		{
			upsampling.push_front(upsamplingBlock(featureChannels * (1 << (level + 1)), featureChannels * (1 << level)));
			register_module("upsampling" + to_string(level), upsampling.front());

			expanding.push_front(levelBlock(featureChannels * (1 << level) * 2, featureChannels * (1 << level)));
			register_module("expandingBlock" + to_string(level), expanding.front());
		}

		output = Conv2d(Conv2dOptions(featureChannels, outChannels, 1));
		seqOutput = Sequential(output);
		register_module("output", output);
	}

	Tensor forward(const Tensor& inputTensor)
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
			downsamplingTensor[level] = downsampling[level]->forward(contractingTensor[level]);
		}

		bottleneckTensor = bottleneck->forward(downsamplingTensor.back());

		for (int level = levels - 2; level >= 0; level--)
		{
			upsamplingTensor[level] = upsampling[level]->forward(level == levels - 2 ? bottleneckTensor : expandingTensor[level + 1]);
			if (paddingSize == 0)
			{ //apply cropping to the contracting tensor in order to concatenate with the same-level expanding tensor
				int oldXSize = contractingTensor[level].size(2);
				int oldYSize = contractingTensor[level].size(3);
				int newXSize = upsamplingTensor[level].size(2);
				int newYSize = upsamplingTensor[level].size(3);
				int startX = oldXSize / 2 - newXSize / 2;
				int startY = oldYSize / 2 - newYSize / 2;
				contractingTensor[level] = contractingTensor[level].slice(2, startX, startX + newXSize);
				contractingTensor[level] = contractingTensor[level].slice(3, startY, startY + newYSize);
			}
			expandingTensor[level] = expanding[level]->forward(cat({ contractingTensor[level],upsamplingTensor[level] }, 1));
		}

		outputTensor = output->forward(expandingTensor.front());

		if (showSizes)
		{
			cout << "input:  " << inputTensor.sizes() << endl;
			for (int level = 0; level < levels - 1; level++)
			{
				for (int i = 0; i < level; i++) cout << " "; cout << " contracting" << level << ":  " << contractingTensor[level].sizes() << endl;
				for (int i = 0; i < level; i++) cout << " "; cout << " downsampling" << level << ": " << downsamplingTensor[level].sizes() << endl;
			}
			for (int i = 0; i < levels - 1; i++) cout << " "; cout << " bottleneck:    " << bottleneckTensor.sizes() << endl;
			for (int level = levels - 2; level >= 0; level--)
			{
				for (int i = 0; i < level; i++) cout << " "; cout << " upsampling" << level << ":   " << upsamplingTensor[level].sizes() << endl;
				for (int i = 0; i < level; i++) cout << " "; cout << " expanding" << level << ":    " << expandingTensor[level].sizes() << endl;
			}
			cout << "output: " << outputTensor.sizes() << endl;
			showSizes = false;
		}

		return outputTensor;
	}

	//the 2d tensor size you pass to the model must be a multiple of this
	int sizeMultiple() { return 1 << (levels - 1); }
private:
	Sequential levelBlock(int inChannels, int outChannels)
	{
		if (batchNorm)
		{
			if (partialConvolution)
				return Sequential(
					PartialConv2d(Conv2dOptions(inChannels, outChannels, kernelSize).padding(paddingSize)),
					BatchNorm2d(outChannels),
					ReLU(),
					PartialConv2d(Conv2dOptions(outChannels, outChannels, kernelSize).padding(paddingSize)),
					BatchNorm2d(outChannels),
					ReLU()
				);
			else
				return Sequential(
					Conv2d(Conv2dOptions(inChannels, outChannels, kernelSize).padding(paddingSize)),
					BatchNorm2d(outChannels),
					ReLU(),
					Conv2d(Conv2dOptions(outChannels, outChannels, kernelSize).padding(paddingSize)),
					BatchNorm2d(outChannels),
					ReLU()
				);
		}
		else
		{
			if (partialConvolution)
				return Sequential(
					PartialConv2d(Conv2dOptions(inChannels, outChannels, kernelSize).padding(paddingSize)),
					ReLU(),
					PartialConv2d(Conv2dOptions(outChannels, outChannels, kernelSize).padding(paddingSize)),
					ReLU()
				);
			else
				return Sequential(
					Conv2d(Conv2dOptions(inChannels, outChannels, kernelSize).padding(paddingSize)),
					ReLU(),
					Conv2d(Conv2dOptions(outChannels, outChannels, kernelSize).padding(paddingSize)),
					ReLU()
				);
		}
	}

	Sequential downsamplingBlock(int channels)
	{
		if (convolutionDownsampling)
		{
			if (partialConvolution)
				return Sequential(
					PartialConv2d(Conv2dOptions(channels, channels, kernelSize).stride(2).padding(paddingSize))
				);
			else
				return Sequential(
					Conv2d(Conv2dOptions(channels, channels, kernelSize).stride(2).padding(paddingSize))
				);
		}
		else
		{
			return Sequential(
				MaxPool2d(MaxPool2dOptions(2).stride(2))
			);
		}
	}

	Sequential upsamplingBlock(int inChannels, int outChannels)
	{
		if (convolutionUpsampling)
		{
			if (partialConvolution)
				return Sequential(
					Upsample(UpsampleOptions().scale_factor(vector<double>({ 2, 2 })).mode(kNearest)),
					PartialConv2d(Conv2dOptions(inChannels, outChannels, kernelSize).padding(paddingSize))
				);
			else
				return Sequential(
					Upsample(UpsampleOptions().scale_factor(vector<double>({ 2, 2 })).mode(kNearest)),
					Conv2d(Conv2dOptions(inChannels, outChannels, kernelSize).padding(paddingSize))
				);
		}
		else
		{
			return Sequential(
				ConvTranspose2d(ConvTranspose2dOptions(inChannels, outChannels, 2).stride(2))
			);
		}
	}

	void initialize(shared_ptr<MyContainer> module, string name, bool highPriority) override;
	void analyzeOperations(int warmup, int repeat, bool isWcet) override;
	Tensor forwardDummy(Tensor input, MyStream* str) override;
	Tensor releaseOperations(Tensor input) override;
};

shared_ptr<UNet> unet(int numClasses);