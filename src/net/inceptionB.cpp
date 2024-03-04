# include "inception.hpp"

using namespace std;
using namespace FGPRS;
using namespace torch;
using namespace nn;

void InceptionBImpl::initialize(shared_ptr<MyContainer> module, string name, bool highPriority)
{
	moduleName = name;
	this->highPriority = highPriority;

	_oBranch3x3 = make_shared<Operation>("branch3x3", module, (Sequential(branch3x3)).ptr(), false);
	_oBranch3x3dbl = make_shared<Operation>("branch3x3dbl", module,
		(Sequential(branch3x3dbl_1, branch3x3dbl_2)).ptr(), false);
	_oBranchPool = make_shared<Operation>("branchPool", module, (Sequential(branch_pool)).ptr(), false);

	addOperation(_oBranch3x3);
	addOperation(_oBranch3x3dbl);
	addOperation(_oBranchPool);
}