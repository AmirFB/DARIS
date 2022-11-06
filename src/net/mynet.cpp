# include <mynet.h>

using namespace FGPRS;

MyNet::MyNet() : Container() {}
	
MyNet::MyNet(int size)
{
	inputSize = size;

	lin1 = register_module("lin1", Linear(inputSize, inputSize * 2));
	lin2 = register_module("lin2", Linear(inputSize * 2, inputSize * 4));
	// lin3 = register_module("lin3", Linear(inputSize * 4, inputSize * 5));
	// lin4 = register_module("lin4", Linear(inputSize * 5, inputSize));
	// lin5 = register_module("lin5", Linear(inputSize * 6, inputSize));
}

void MyNet::setSize(int size)
{
	inputSize = size;

	lin1 = register_module("lin1", Linear(inputSize, inputSize * 2));
	lin2 = register_module("lin2", Linear(inputSize * 2, inputSize * 4));
}

Tensor MyNet::_forward_impl(Tensor x)
{
	auto y = lin1->forward(x);
	y = lin2->forward(y);
	// y = lin3->forward(y);
	// y = lin4->forward(y);
	// y = lin5->forward(y);

	return y;
}

Tensor MyNet::forward(Tensor x)
{
	return _forward_impl(x);
}