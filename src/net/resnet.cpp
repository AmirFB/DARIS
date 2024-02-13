# include "resnet.hpp"

using namespace FGPRS;

void ResNet::initialize(shared_ptr<MyContainer> module, string name, bool highPriority)
{
	moduleName = name;
	this->highPriority = highPriority;

	_op1 = make_shared<Operation>("layer1", module, _layer1.ptr(), false);
	_op2 = make_shared<Operation>("layer2", module, _layer2.ptr(), false);
	_op3 = make_shared<Operation>("layer3", module, _layer3.ptr(), false);
	_op4 = make_shared<Operation>("layer4", module, _layer4.ptr(), true);

	addOperation(_op1);
	addOperation(_op2);
	addOperation(_op3);
	addOperation(_op4);
}

shared_ptr<ResNet> resnet18(int numClasses)
{
	auto model = make_shared<ResNet>(ResNet({ 64, 128, 256, 512 }, 18, { 2, 2, 2, 2 }, numClasses));
	return model;
}