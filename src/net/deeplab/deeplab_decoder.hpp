# pragma once
# include"../utils/util.h"

//struct StackSequentailImpl : torch::nn::SequentialImpl {
//	using SequentialImpl::SequentialImpl;
//
//	torch::Tensor forward(torch::Tensor x) {
//		return SequentialImpl::forward(x);
//	}
//}; TORCH_MODULE(StackSequentail);

torch::nn::Sequential ASPPConv(int in_channels, int out_channels, int dilation);

torch::nn::Sequential SeparableConv2d(int in_channels, int out_channels, int kernel_size, int stride = 1,
	int padding = 0, int dilation = 1, bool bias = true);

torch::nn::Sequential ASPPSeparableConv(int in_channels, int out_channels, int dilation);

class ASPPPoolingImpl : public torch::nn::Module
{
public:
	torch::nn::Sequential seq{ nullptr };
	ASPPPoolingImpl(int in_channels, int out_channels);
	torch::Tensor forward(torch::Tensor x);

}; TORCH_MODULE(ASPPPooling);

class ASPPImpl : public torch::nn::Module
{
public:
	ASPPImpl(int in_channels, int out_channels, vector<int> atrous_rates, bool separable = false);
	torch::Tensor forward(torch::Tensor x);
private:
	torch::nn::ModuleList modules{};
	ASPPPooling aspppooling{ nullptr };
	torch::nn::Sequential project{ nullptr };
}; TORCH_MODULE(ASPP);

class DeepLabV3DecoderImpl : public torch::nn::Module
{
public:
	DeepLabV3DecoderImpl(int in_channels, int out_channels = 256, vector<int> atrous_rates = { 12, 24, 36 });
	torch::Tensor forward(vector< torch::Tensor> x);
	int out_channels = 0;
private:
	torch::nn::Sequential seq{};
}; TORCH_MODULE(DeepLabV3Decoder);

class DeepLabV3PlusDecoderImpl :public torch::nn::Module
{
public:
	DeepLabV3PlusDecoderImpl(vector<int> encoder_channels, int out_channels,
		vector<int> atrous_rates, int output_stride = 16);
	torch::Tensor forward(vector< torch::Tensor> x);
private:
	ASPP aspp{ nullptr };
	torch::nn::Sequential aspp_seq{ nullptr };
	torch::nn::Upsample up{ nullptr };
	torch::nn::Sequential block1{ nullptr };
	torch::nn::Sequential block2{ nullptr };
}; TORCH_MODULE(DeepLabV3PlusDecoder);