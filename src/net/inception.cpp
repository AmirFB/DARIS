# include "inception.hpp"

using namespace std;
using namespace FGPRS;
using namespace torch;
using namespace nn;

void Inception3::initialize(shared_ptr<MyContainer> module)
{
	_oLayer0 = make_shared<Operation>("layer0", module, layer0.ptr(), false);

	_oLayerA1 = make_shared<Operation>("layerA1", module, layerA1, false, 4);
	_oLayerA2 = make_shared<Operation>("layerA2", module, layerA2, false, 4);
	_oLayerA3 = make_shared<Operation>("layerA3", module, layerA3, false, 4);
	// _oLayerA1 = make_shared<Operation>("layerA1", module, (Sequential(layerA1)).ptr(), false);
	// _oLayerA2 = make_shared<Operation>("layerA2", module, (Sequential(layerA2)).ptr(), false);
	// _oLayerA3 = make_shared<Operation>("layerA3", module, (Sequential(layerA3)).ptr(), false);

	// _oLayerB = make_shared<Operation>("layerB", module, layerB, false, 2);
	_oLayerB = make_shared<Operation>("layerB", module, (Sequential(layerB)).ptr(), false);

	// _oLayerC1 = make_shared<Operation>("layerC1", module, layerC1, false, 3);
	// _oLayerC2 = make_shared<Operation>("layerC2", module, layerC2, false, 3);
	// _oLayerC3 = make_shared<Operation>("layerC3", module, layerC3, false, 3);
	// _oLayerC4 = make_shared<Operation>("layerC4", module, layerC4, false, 3);
	_oLayerC1 = make_shared<Operation>("layerC1", module, (Sequential(layerC1)).ptr(), false);
	_oLayerC2 = make_shared<Operation>("layerC2", module, (Sequential(layerC2)).ptr(), false);
	_oLayerC3 = make_shared<Operation>("layerC3", module, (Sequential(layerC3)).ptr(), false);
	_oLayerC4 = make_shared<Operation>("layerC4", module, (Sequential(layerC4)).ptr(), false);

	_oLayerD = make_shared<Operation>("layerD", module, (Sequential(layerD)).ptr(), false);

	_oLayerE1 = make_shared<Operation>("layerE1", module, (Sequential(layerE1)).ptr(), false);
	_oLayerE2 = make_shared<Operation>("layerE2", module, (Sequential(layerE2)).ptr(), false);

	_oLayerX = make_shared<Operation>("layerX", module, layerX.ptr(), false);

	addOperation(_oLayer0);

	addOperation(_oLayerA1);
	addOperation(_oLayerA2);
	addOperation(_oLayerA3);

	addOperation(_oLayerB);

	addOperation(_oLayerC1);
	addOperation(_oLayerC2);
	addOperation(_oLayerC3);
	addOperation(_oLayerC4);

	addOperation(_oLayerD);

	addOperation(_oLayerE1);
	addOperation(_oLayerE2);

	addOperation(_oLayerX);
}

shared_ptr<Inception3> inception3(int numClasses)
{
	auto model = make_shared<Inception3>(Inception3(numClasses));
	return model;
}