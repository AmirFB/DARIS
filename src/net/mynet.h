# ifndef __MYNET__
# define __MYNET__

# include <torch/torch.h>

# include <schd.h>
# include <container.h>

using namespace std;
using namespace torch;
using namespace torch::nn;
using namespace FGPRS;

namespace FGPRS
{
	struct MyNet : Container
	{
		public:
		Linear lin1{nullptr}, lin2{nullptr}, lin3{nullptr}, lin4{nullptr}, lin5{nullptr};
		BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr}, bn4{nullptr}, bn5{nullptr};
		ReLU relu1{nullptr}, relu2{nullptr}, relu3{nullptr}, relu4{nullptr}, relu5{nullptr};
		MaxPool2d maxpool1{nullptr}, maxpool2{nullptr}, maxpool3{nullptr}, maxpool4{nullptr}, maxpool5{nullptr};

		int inputSize;

		MyNet();
		MyNet(int size);
		void setSize(int size);
		Tensor _forward_impl(Tensor x);

		Tensor forward(Tensor x);
	};
}

# endif