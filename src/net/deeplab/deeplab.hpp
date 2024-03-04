# pragma once

# include "resnet_encoder.hpp"
# include "vgg_encoder.hpp"
# include "deeplab_decoder.hpp"

# include <cnt.hpp>

using namespace std;
using namespace FGPRS;
using namespace torch;
using namespace nn;

class DeepLabV3PlusImpl : public MyContainer
{
public:
	DeepLabV3PlusImpl() {}

	DeepLabV3PlusImpl(int _num_classes, int encoder_depth,
		int encoder_output_stride, int decoder_channels, int in_channels, double upsampling)
	{
		num_classes = _num_classes;
		auto encoder_param = encoder_params();
		vector<int> encoder_channels = encoder_param[encoder_name]["out_channels"];

		encoder = new resnet18(1000);

		if (encoder_output_stride == 8)
			encoder->make_dilated({ 5,4 }, { 4,2 });

		else if (encoder_output_stride == 16)
			encoder->make_dilated({ 5 }, { 2 });

		// m_decoder = DeepLabV3PlusDecoder(encoder_channels, decoder_channels, decoder_atrous_rates, encoder_output_stride);
		// m_head = SegmentationHead(decoder_channels, num_classes, 1, upsampling);

		// m_encoder = *register_module("encoder", shared_ptr<Backbone>(encoder));
		// register_module("encoder", shared_ptr<Backbone>(encoder));
		m_encoder = register_module("encoder", shared_ptr<ResNet>(encoder));
		m_decoder = register_module("decoder", DeepLabV3PlusDecoder(encoder_channels, decoder_channels, decoder_atrous_rates, encoder_output_stride));
		m_head = register_module("head", SegmentationHead(decoder_channels, num_classes, 1, upsampling));
	}

	Tensor forward(Tensor x);
private:
	ResNet* encoder;
	shared_ptr<ResNet> m_encoder;
	DeepLabV3PlusDecoder m_decoder{ nullptr };
	SegmentationHead m_head{ nullptr };
	shared_ptr<Operation> o_head;

	int num_classes = 1;
	vector<int> decoder_atrous_rates = { 12, 24, 36 };

public:
	void assignOperations() override;
	Tensor schedule(Tensor input, int level) override;
	Tensor analyze(int warmup, int repeat, Tensor input, int index, int level) override;
	double assignExecutionTime(int level, int contextIndex, double executionTimeStack) override;
	double assignDeadline(double quota, int level, int contextIndex, double deadlineStack) override;
	void setAbsoluteDeadline(int level, steady_clock::time_point start, int bias) override;
}; TORCH_MODULE(DeepLabV3Plus);