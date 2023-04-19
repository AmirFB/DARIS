# pragma once
# include "util.hpp"

# include <sqnl.hpp>

using namespace torch;
using namespace nn;
using namespace FGPRS;

MySequential ASPPConv(int in_channels, int out_channels, int dilation);

MySequential SeparableConv2d(int in_channels, int out_channels, int kernel_size, int stride = 1,
	int padding = 0, int dilation = 1, bool bias = true);

MySequential ASPPSeparableConv(int in_channels, int out_channels, int dilation);

class ASPPPoolingImpl : public MyContainer
{
public:
	MySequential seq{ nullptr };
	ASPPPoolingImpl(int in_channels, int out_channels);
	torch::Tensor forward(torch::Tensor x);

}; TORCH_MODULE(ASPPPooling);

class ASPPImpl : public MyContainer
{
public:
	ASPPImpl(int in_channels, int out_channels, vector<int> atrous_rates, bool separable = false);
	torch::Tensor forward(torch::Tensor x);
private:
	vector<MySequential> modules;
	ASPPPooling aspppooling{ nullptr };
	MySequential project{ nullptr };
}; TORCH_MODULE(ASPP);

class DeepLabV3DecoderImpl : public MyContainer
{
public:
	DeepLabV3DecoderImpl(int in_channels, int out_channels = 256, vector<int> atrous_rates = { 12, 24, 36 });
	torch::Tensor forward(vector< torch::Tensor> x);
	int out_channels = 0;
private:
	MySequential seq{};
}; TORCH_MODULE(DeepLabV3Decoder);

class DeepLabV3PlusDecoderImpl :public MyContainer
{
public:
	DeepLabV3PlusDecoderImpl(vector<int> encoder_channels, int out_channels,
		vector<int> atrous_rates, int output_stride = 16);
	torch::Tensor forward(vector< torch::Tensor> x);
private:
	ASPP aspp{ nullptr };
	MySequential aspp_seq{ nullptr };
	Upsample up{ nullptr };
	MySequential block1{ nullptr };
	MySequential block2{ nullptr };
}; TORCH_MODULE(DeepLabV3PlusDecoder);