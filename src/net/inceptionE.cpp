# include "inception.hpp"

using namespace std;
using namespace FGPRS;
using namespace torch;
using namespace nn;

void InceptionEImpl::initialize(shared_ptr<MyContainer> module, string name, bool highPriority)
{
	moduleName = name;
	this->highPriority = highPriority;

	_oBranch1x1 = make_shared<Operation>("branch1x1", module, (Sequential(branch1x1)).ptr(), false);
	_oBranch3x3 = make_shared<Operation>("branch3x3", module,
		(Sequential(branch3x3_1, branch3x3_2.ptr())).ptr(), false);
	_oBranch3x3dbl = make_shared<Operation>("branch3x3dbl", module,
		(Sequential(branch3x3dbl_1, branch3x3dbl_2, branch3x3dbl_3)).ptr(), false);
	_oBranchPool = make_shared<Operation>("branchPool", module,
		(Sequential(branch_pool_1, branch_pool_2)).ptr(), false);

	addOperation(_oBranch1x1);
	addOperation(_oBranch3x3);
	addOperation(_oBranch3x3dbl);
	addOperation(_oBranchPool);
}