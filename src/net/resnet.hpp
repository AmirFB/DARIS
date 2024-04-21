# pragma once

# include "cnt.hpp"

# include <torch/torch.h>
# include <vector>

using namespace std;
using namespace torch;
using namespace torch::nn;
using namespace FGPRS;

struct SimpleBlock : Module
{
	Conv2d _conv1{ nullptr }, _conv2{ nullptr };
	BatchNorm2d _bn1{ nullptr }, _bn2{ nullptr };
	Sequential _downsample{ nullptr };

	SimpleBlock(int64_t in_channels, int64_t out_channels, int64_t stride = 1, Sequential downsample = Sequential{ nullptr })
	{
		_conv1 = register_module("conv1",
			Conv2d(Conv2dOptions(in_channels, out_channels, 3).stride(stride).padding(1).bias(false)));
		_bn1 = register_module("bn1", BatchNorm2d(out_channels));
		_conv2 = register_module("conv2",
			Conv2d(Conv2dOptions(out_channels, out_channels, 3).stride(1).padding(1).bias(false)));
		_bn2 = register_module("bn2", BatchNorm2d(out_channels));

		if (!downsample.is_empty())
			_downsample = register_module("downsample", downsample);
	}

	Tensor forward(Tensor x)
	{
		Tensor residual = x;

		x = relu(_bn1(_conv1(x)));
		x = _bn2(_conv2(x));

		if (!_downsample.is_empty())
			residual = _downsample->forward(residual);

		x += residual;
		x = relu(x);

		return x;
	}
};

struct BottleneckBlock : Module
{
	Conv2d _conv1{ nullptr }, _conv2{ nullptr }, _conv3{ nullptr };
	BatchNorm2d _bn1{ nullptr }, _bn2{ nullptr }, _bn3{ nullptr };
	Sequential _downsample{ nullptr };

	BottleneckBlock(int64_t in_channels, int64_t out_channels, int64_t stride = 1, Sequential downsample = Sequential())
	{
		_conv1 = register_module("conv1",
			Conv2d(Conv2dOptions(in_channels, out_channels / 4, 1).bias(false)));
		_bn1 = register_module("bn1", BatchNorm2d(out_channels / 4));
		_conv2 = register_module("conv2",
			Conv2d(Conv2dOptions(out_channels / 4, out_channels / 4, 3).stride(stride).padding(1).bias(false)));
		_bn2 = register_module("bn2", BatchNorm2d(out_channels / 4));
		_conv3 = register_module("conv3",
			Conv2d(Conv2dOptions(out_channels / 4, out_channels, 1).bias(false)));
		_bn3 = register_module("bn3", BatchNorm2d(out_channels));

		if (!downsample.is_empty())
			_downsample = register_module("downsample", downsample);
	}

	Tensor forward(Tensor x)
	{
		Tensor residual = x;

		x = relu(_bn1(_conv1(x)));
		x = relu(_bn2(_conv2(x)));
		x = _bn3(_conv3(x));

		// if (!_downsample.is_empty())
		if (x.sizes()[1] != residual.sizes()[1])
			residual = _downsample->forward(residual);

		x += residual;
		x = relu(x);

		return x;
	}
};

struct FCSoftmaxModule : Module
{
	Linear fc{ nullptr };

	FCSoftmaxModule(int64_t in_features, int64_t num_classes)
	{
		fc = register_module("fc", Linear(in_features, num_classes));
	}

	Tensor forward(Tensor x)
	{
		x = adaptive_avg_pool2d(x, { 1, 1 });
		x = x.view({ x.size(0), -1 });
		x = fc(x);
		return log_softmax(x, 1);
	}
};

struct ResNet : MyContainer
{
private:
	int64_t in_channels{ 3 };
	int64_t _numClasses{ 1000 };
	Sequential _layer1{ nullptr }, _layer2{ nullptr }, _layer3{ nullptr }, _layer4{ nullptr };
	shared_ptr<Operation> _op1, _op2, _op3, _op4;

public:
	ResNet(const vector<int>& layer_sizes, int block_type, const vector<int>& num_blocks, int numClasses)
		: _numClasses(numClasses)
	{
		auto layer1 = Sequential(
			Conv2d(Conv2dOptions(in_channels, 64, 7).stride(2).padding(3).bias(false)),
			BatchNorm2d(64),
			ReLU(),
			MaxPool2d(MaxPool2dOptions(3).stride(2).padding(1))
		);

		make_layer(layer1, layer_sizes[0], block_type, num_blocks[0], 1, true);
		_layer1 = register_module("layer1", layer1);

		auto layer2 = Sequential();
		make_layer(layer2, layer_sizes[1], block_type, num_blocks[1], 2);
		_layer2 = register_module("layer2", layer2);

		auto layer3 = Sequential();
		make_layer(layer3, layer_sizes[2], block_type, num_blocks[2], 2);
		_layer3 = register_module("layer3", layer3);

		auto layer4 = Sequential();
		make_layer(layer4, layer_sizes[3], block_type, num_blocks[3], 2);
		layer4->push_back(FCSoftmaxModule(layer_sizes[3], _numClasses));
		_layer4 = register_module("layer4", layer4);
	}

private:
	Sequential make_layer(Sequential& layer, int64_t channels, int block_type, int num_blocks, int64_t stride = 1, bool first = false)
	{
		auto downsample = Sequential{ nullptr };
		int64_t in_channels = first ? 64 : channels / 2;

		if (stride != 1 || !first || block_type >= 50)
			downsample = Sequential(
				Conv2d(Conv2dOptions(in_channels, channels, 1).stride(stride).bias(false)),
				BatchNorm2d(channels)
			);

		if (block_type == 18 || block_type == 34)
			layer->push_back(SimpleBlock(in_channels, channels, stride, downsample));

		else
			layer->push_back(BottleneckBlock(in_channels, channels, stride, downsample));

		in_channels = channels;
		stride = 1;

		for (int i = 0; i < (num_blocks - 1); ++i)
		{
			if (block_type == 18 || block_type == 34)
				layer->push_back(SimpleBlock(in_channels, channels, stride));

			else
				layer->push_back(BottleneckBlock(in_channels, channels, stride));
		}

		return layer;
	}

public:
	Tensor forward(Tensor x)
	{
		x = _layer1->forward(x);
		x = _layer2->forward(x);
		x = _layer3->forward(x);
		x = _layer4->forward(x);

		return x;
	}

	void initialize(shared_ptr<MyContainer> module) override;
};

shared_ptr<ResNet> resnet18(int numClasses);
shared_ptr<ResNet> resnet50(int numClasses);