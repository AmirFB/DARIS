# include <cnt.hpp>

# include <torch/torch.h>

using namespace std;
using namespace torch;
using namespace nn;

namespace FGPRS
{
	class BasicConv2dImpl : public MyContainer
	{
	private:
		Conv2d conv = nullptr;
		BatchNorm2d bn = nullptr;

	public:
		BasicConv2dImpl(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0, int bias = false)
		{
			conv = register_module("conv",
				Conv2d(Conv2dOptions(in_channels, out_channels, kernel_size).stride(stride).padding(padding).bias(bias)));
			bn = register_module("bn", BatchNorm2d(out_channels));
		}

		BasicConv2dImpl(int in_channels, int out_channels, IntArrayRef kernel_size, int stride, IntArrayRef padding)
		{
			conv = register_module("conv",
				Conv2d(Conv2dOptions(in_channels, out_channels, kernel_size).stride(stride).padding(padding).bias(false)));
			bn = register_module("bn", BatchNorm2d(out_channels));
		}

		Tensor forward(Tensor x)
		{
			x = conv->forward(x);
			x = bn->forward(x);
			return relu(x);
		}
	};
	TORCH_MODULE(BasicConv2d);

	class InceptionAImpl : public MyContainer
	{
	private:
		BasicConv2d  branch1x1 = nullptr, branch5x5_1 = nullptr, branch5x5_2 = nullptr,
			branch3x3dbl_1 = nullptr, branch3x3dbl_2 = nullptr, branch3x3dbl_3 = nullptr,
			branch_pool_2 = nullptr;
		AvgPool2d branch_pool_1 = nullptr;
		shared_ptr<Operation> _oBranch1x1, _oBranch5x5, _oBranch3x3dbl, _oBranchPool;

	public:
		InceptionAImpl(int in_channels, int pool_features)
		{
			branch1x1 = register_module("branch1x1", BasicConv2d(in_channels, 64, 1));

			branch5x5_1 = register_module("branch5x5_1", BasicConv2d(in_channels, 48, 1, 1, 0));
			branch5x5_2 = register_module("branch5x5_2", BasicConv2d(48, 64, 5, 1, 2));

			branch3x3dbl_1 = register_module("branch3x3dbl_1", BasicConv2d(in_channels, 64, 1, 1, 0));
			branch3x3dbl_2 = register_module("branch3x3dbl_2", BasicConv2d(64, 96, 3, 1, 1));
			branch3x3dbl_3 = register_module("branch3x3dbl_3", BasicConv2d(96, 96, 3, 1, 1));

			branch_pool_1 = register_module("branch_pool_1", AvgPool2d(AvgPool2dOptions(3).stride(1).padding(1)));
			branch_pool_2 = register_module("branch_pool_2", BasicConv2d(in_channels, pool_features, 1, 1, 0));
		}

		Tensor forward(Tensor x)
		{
			Tensor branch1x1 = this->branch1x1->forward(x);

			Tensor branch5x5 = this->branch5x5_1->forward(x);
			branch5x5 = this->branch5x5_2->forward(branch5x5);

			Tensor branch3x3dbl = this->branch3x3dbl_1->forward(x);
			branch3x3dbl = this->branch3x3dbl_2->forward(branch3x3dbl);
			branch3x3dbl = this->branch3x3dbl_3->forward(branch3x3dbl);

			Tensor branch_pool = this->branch_pool_1->forward(x);
			branch_pool = this->branch_pool_2->forward(branch_pool);

			return cat({ branch1x1, branch5x5, branch3x3dbl, branch_pool }, 1);
		}

		void initialize(shared_ptr<MyContainer> module, string name, bool highPriority);
	};
	TORCH_MODULE(InceptionA);

	class InceptionBImpl : public MyContainer
	{
	private:
		BasicConv2d branch3x3 = nullptr, branch3x3dbl_1 = nullptr, branch3x3dbl_2 = nullptr,
			branch3x3dbl_3 = nullptr;
		MaxPool2d branch_pool = nullptr;
		shared_ptr<Operation> _oBranch3x3, _oBranch3x3dbl, _oBranchPool;

	public:
		InceptionBImpl(int in_channels)
		{
			branch3x3 = register_module("branch3x3", BasicConv2d(in_channels, 384, 3, 2, 0));

			branch3x3dbl_1 = register_module("branch3x3dbl_1", BasicConv2d(in_channels, 64, 1, 1, 0));
			branch3x3dbl_2 = register_module("branch3x3dbl_2", BasicConv2d(64, 96, 3, 1, 1));
			branch3x3dbl_3 = register_module("branch3x3dbl_3", BasicConv2d(96, 96, 3, 2, 0));

			branch_pool = register_module("branch_pool", MaxPool2d(MaxPool2dOptions(3).stride(2).padding(0)));
		}

		Tensor forward(Tensor x)
		{
			Tensor branch3x3 = this->branch3x3->forward(x);

			Tensor branch3x3dbl = this->branch3x3dbl_1->forward(x);
			branch3x3dbl = this->branch3x3dbl_2->forward(branch3x3dbl);
			branch3x3dbl = this->branch3x3dbl_3->forward(branch3x3dbl);

			Tensor branch_pool = this->branch_pool->forward(x);

			return cat({ branch3x3, branch3x3dbl, branch_pool }, 1);
		}

		void initialize(shared_ptr<MyContainer> module, string name, bool highPriority);
	};
	TORCH_MODULE(InceptionB);

	class InceptionCImpl : public MyContainer
	{
	private:
		BasicConv2d branch1x1 = nullptr, branch7x7_1 = nullptr, branch7x7_2 = nullptr, branch7x7_3 = nullptr,
			branch7x7dbl_1 = nullptr, branch7x7dbl_2 = nullptr, branch7x7dbl_3 = nullptr,
			branch7x7dbl_4 = nullptr, branch7x7dbl_5 = nullptr, branch_pool_2 = nullptr;
		AvgPool2d branch_pool_1 = nullptr;
		shared_ptr<Operation> _oBranch1x1, _oBranch7x7, _oBranch7x7dbl, _oBranchPool;

	public:
		InceptionCImpl(int in_channels, int channels_7x7)
		{
			branch1x1 = register_module("branch1x1", BasicConv2d(in_channels, 192, 1, 1, 0));

			branch7x7_1 = register_module("branch7x7_1",
				BasicConv2d(in_channels, channels_7x7, 1, 1, 0));
			branch7x7_2 = register_module("branch7x7_2",
				BasicConv2d(channels_7x7, channels_7x7, IntArrayRef({ 1, 7 }), 1, IntArrayRef({ 0, 3 })));
			branch7x7_3 = register_module("branch7x7_3",
				BasicConv2d(channels_7x7, 192, IntArrayRef({ 7, 1 }), 1, IntArrayRef({ 3, 0 })));

			branch7x7dbl_1 = register_module("branch7x7dbl_1",
				BasicConv2d(in_channels, channels_7x7, 1, 1, 0));
			branch7x7dbl_2 = register_module("branch7x7dbl_2",
				BasicConv2d(channels_7x7, channels_7x7, IntArrayRef({ 7, 1 }), 1, IntArrayRef({ 3, 0 })));
			branch7x7dbl_3 = register_module("branch7x7dbl_3",
				BasicConv2d(channels_7x7, channels_7x7, IntArrayRef({ 1, 7 }), 1, IntArrayRef({ 0, 3 })));
			branch7x7dbl_4 = register_module("branch7x7dbl_4",
				BasicConv2d(channels_7x7, channels_7x7, IntArrayRef({ 7, 1 }), 1, IntArrayRef({ 3, 0 })));
			branch7x7dbl_5 = register_module("branch7x7dbl_5",
				BasicConv2d(channels_7x7, 192, IntArrayRef({ 1, 7 }), 1, IntArrayRef({ 0, 3 })));

			branch_pool_1 = register_module("branch_pool_1", AvgPool2d(AvgPool2dOptions(3).stride(1).padding(1)));
			branch_pool_2 = register_module("branch_pool_2", BasicConv2d(in_channels, 192, 1, 1, 0));
		}

		Tensor forward(Tensor x)
		{
			Tensor branch1x1 = this->branch1x1->forward(x);

			Tensor branch7x7 = this->branch7x7_1->forward(x);
			branch7x7 = this->branch7x7_2->forward(branch7x7);
			branch7x7 = this->branch7x7_3->forward(branch7x7);

			Tensor branch7x7dbl = this->branch7x7dbl_1->forward(x);
			branch7x7dbl = this->branch7x7dbl_2->forward(branch7x7dbl);
			branch7x7dbl = this->branch7x7dbl_3->forward(branch7x7dbl);
			branch7x7dbl = this->branch7x7dbl_4->forward(branch7x7dbl);
			branch7x7dbl = this->branch7x7dbl_5->forward(branch7x7dbl);

			Tensor branch_pool = this->branch_pool_1(x);
			branch_pool = this->branch_pool_2->forward(branch_pool);

			return cat({ branch1x1, branch7x7, branch7x7dbl, branch_pool }, 1);
		}

		void initialize(shared_ptr<MyContainer> module, string name, bool highPriority);
	};
	TORCH_MODULE(InceptionC);

	class InceptionDImpl : public MyContainer
	{
	private:
		BasicConv2d branch3x3_1 = nullptr, branch3x3_2 = nullptr,
			branch7x7x3_1 = nullptr, branch7x7x3_2 = nullptr, branch7x7x3_3 = nullptr, branch7x7x3_4 = nullptr;
		MaxPool2d branch_pool = nullptr;
		shared_ptr<Operation> _oBranch3x3, _oBranch7x7x3, _oBranchPool;

	public:
		InceptionDImpl(int in_channels)
		{
			branch3x3_1 = register_module("branch3x3_1", BasicConv2d(in_channels, 192, 1, 1, 0));
			branch3x3_2 = register_module("branch3x3_2", BasicConv2d(192, 320, 3, 2, 0));

			branch7x7x3_1 = register_module("branch7x7x3_1", BasicConv2d(in_channels, 192, 1, 1, 0));
			branch7x7x3_2 = register_module("branch7x7x3_2", BasicConv2d(192, 192, IntArrayRef({ 1, 7 }), 1, IntArrayRef({ 0, 3 })));
			branch7x7x3_3 = register_module("branch7x7x3_3", BasicConv2d(192, 192, IntArrayRef({ 7, 1 }), 1, IntArrayRef({ 3, 0 })));
			branch7x7x3_4 = register_module("branch7x7x3_4", BasicConv2d(192, 192, 3, 2, 0));

			branch_pool = register_module("max_pool", MaxPool2d(MaxPool2dOptions(3).stride(2)));
		}

		Tensor forward(Tensor x)
		{
			Tensor branch3x3 = this->branch3x3_1->forward(x);
			branch3x3 = this->branch3x3_2->forward(branch3x3);

			Tensor branch7x7x3 = this->branch7x7x3_1->forward(x);
			branch7x7x3 = this->branch7x7x3_2->forward(branch7x7x3);
			branch7x7x3 = this->branch7x7x3_3->forward(branch7x7x3);
			branch7x7x3 = this->branch7x7x3_4->forward(branch7x7x3);

			Tensor branch_pool = this->branch_pool->forward(x);

			return cat({ branch3x3, branch7x7x3, branch_pool }, 1);
		}

		void initialize(shared_ptr<MyContainer> module, string name, bool highPriority);
	};
	TORCH_MODULE(InceptionD);

	class InceptionEImpl : public MyContainer
	{
	private:
		BasicConv2d branch1x1 = nullptr, branch3x3_1 = nullptr, branch3x3_2a = nullptr, branch3x3_2b = nullptr,
			branch3x3dbl_1 = nullptr, branch3x3dbl_2 = nullptr, branch3x3dbl_3a = nullptr, branch3x3dbl_3b = nullptr,
			branch_pool_2 = nullptr;
		AvgPool2d branch_pool_1 = nullptr;
		Paralleltial branch3x3_2 = nullptr, branch3x3dbl_3 = nullptr;
		shared_ptr<Operation> _oBranch1x1, _oBranch3x3, _oBranch3x3dbl, _oBranchPool;

	public:
		InceptionEImpl(int in_channels)
		{
			branch1x1 = register_module("branch1x1", BasicConv2d(in_channels, 320, 1, 1, 0));

			branch3x3_1 = register_module("branch3x3_1", BasicConv2d(in_channels, 384, 1, 1, 0));
			branch3x3_2a = register_module("branch3x3_2a", BasicConv2d(384, 384, IntArrayRef({ 1, 3 }), 1, IntArrayRef({ 0, 1 })));
			branch3x3_2b = register_module("branch3x3_2b", BasicConv2d(384, 384, IntArrayRef({ 3, 1 }), 1, IntArrayRef({ 1, 0 })));

			vector<shared_ptr<SequentialImpl>> dummy3x3_2;
			dummy3x3_2.push_back((Sequential(branch3x3_2a)).ptr());
			dummy3x3_2.push_back((Sequential(branch3x3_2b)).ptr());
			branch3x3_2 = Paralleltial(dummy3x3_2);

			branch3x3dbl_1 = register_module("branch3x3dbl_1", BasicConv2d(in_channels, 448, 1, 1, 0));
			branch3x3dbl_2 = register_module("branch3x3dbl_2", BasicConv2d(448, 384, IntArrayRef({ 3, 1 }), 1, IntArrayRef({ 1, 0 })));
			branch3x3dbl_3a = register_module("branch3x3dbl_3a", BasicConv2d(384, 384, IntArrayRef({ 1, 3 }), 1, IntArrayRef({ 0, 1 })));
			branch3x3dbl_3b = register_module("branch3x3dbl_3b", BasicConv2d(384, 384, IntArrayRef({ 3, 1 }), 1, IntArrayRef({ 1, 0 })));

			vector<shared_ptr<SequentialImpl>> dummy3x3dbl_3;
			dummy3x3dbl_3.push_back((Sequential(branch3x3dbl_3a)).ptr());
			dummy3x3dbl_3.push_back((Sequential(branch3x3dbl_3b)).ptr());
			branch3x3dbl_3 = Paralleltial(dummy3x3dbl_3);

			branch_pool_1 = register_module("branch_pool_1", AvgPool2d(AvgPool2dOptions(3).stride(1).padding(1)));
			branch_pool_2 = register_module("branch_pool_2", BasicConv2d(in_channels, 192, 1, 1, 0));
		}

		Tensor forward(Tensor x)
		{
			Tensor branch1x1 = this->branch1x1->forward(x);

			Tensor branch3x3 = branch3x3_1->forward(x);
			branch3x3 = cat({ branch3x3_2a->forward(branch3x3),  branch3x3_2b->forward(branch3x3) }, 1);

			Tensor branch3x3dbl = branch3x3dbl_1->forward(x);
			branch3x3dbl = branch3x3dbl_2->forward(branch3x3dbl);
			branch3x3dbl = cat({ branch3x3dbl_3a->forward(branch3x3dbl), branch3x3dbl_3b->forward(branch3x3dbl) }, 1);

			Tensor branch_pool = this->branch_pool_1->forward(x);
			branch_pool = this->branch_pool_2->forward(branch_pool);

			return cat({ branch1x1, branch3x3, branch3x3dbl, branch_pool }, 1);
		}

		void initialize(shared_ptr<MyContainer> module, string name, bool highPriority);
	};
	TORCH_MODULE(InceptionE);

	class Inception3 : public MyContainer
	{
	private:
		Sequential layer0 = nullptr, layerX = nullptr;
		InceptionA layerA1 = nullptr, layerA2 = nullptr, layerA3 = nullptr;
		InceptionB layerB = nullptr;
		InceptionC layerC1 = nullptr, layerC2 = nullptr, layerC3 = nullptr, layerC4 = nullptr;
		InceptionD layerD = nullptr;
		InceptionE layerE1 = nullptr, layerE2 = nullptr;
		shared_ptr<Operation> _oLayer0, _oLayerA1, _oLayerA2, _oLayerA3,
			_oLayerB, _oLayerC1, _oLayerC2, _oLayerC3, _oLayerC4,
			_oLayerD, _oLayerE1, _oLayerE2, _oLayerX;

	public:
		Inception3(int numClasses, double dropout = 0.5)
		{
			layer0 = register_module("layer0", Sequential(
				BasicConv2d(3, 32, 3, 2, 0),
				BasicConv2d(32, 32, 3, 1, 0),
				BasicConv2d(32, 64, 3, 1, 1),
				MaxPool2d(MaxPool2dOptions(3).stride(2)),
				BasicConv2d(64, 80, 1, 1, 0),
				BasicConv2d(80, 192, 3, 1, 0),
				MaxPool2d(MaxPool2dOptions(3).stride(2))
			));

			layerA1 = register_module("layerA1", InceptionA(192, 32));
			layerA2 = register_module("layerA2", InceptionA(256, 64));
			layerA3 = register_module("layerA3", InceptionA(288, 64));

			layerB = register_module("layerB", InceptionB(288));

			layerC1 = register_module("layerC1", InceptionC(768, 128));
			layerC2 = register_module("layerC2", InceptionC(768, 160));
			layerC3 = register_module("layerC3", InceptionC(768, 160));
			layerC4 = register_module("layerC4", InceptionC(768, 192));

			layerD = register_module("layerD", InceptionD(768));

			layerE1 = register_module("layerE1", InceptionE(1280));
			layerE2 = register_module("layerE2", InceptionE(2048));

			layerX = register_module("layerX", Sequential(
				AdaptiveAvgPool2d(1),
				Dropout(dropout),
				Flatten(),
				Linear(2048, numClasses),
				Softmax(1)
			));
		}

		void initialize(shared_ptr<MyContainer> module, string name, bool highPriority);
	};
}