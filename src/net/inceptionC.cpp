# include "inception.hpp"

using namespace std;
using namespace FGPRS;
using namespace torch;
using namespace nn;

void InceptionCImpl::initialize(shared_ptr<MyContainer> module, string name, bool highPriority)
{
	moduleName = name;
	this->highPriority = highPriority;

	_oBranch1x1 = make_shared<Operation>("branch1x1", module, (Sequential(branch1x1)).ptr(), false);
	_oBranch7x7 = make_shared<Operation>("branch7x7", module,
		(Sequential(branch7x7_1, branch7x7_2, branch7x7_2)).ptr(), false);
	_oBranch7x7dbl = make_shared<Operation>("branch7x7dbl", module,
		(Sequential(branch7x7dbl_1, branch7x7dbl_2, branch7x7dbl_3, branch7x7dbl_4, branch7x7dbl_5)).ptr(), false);
	_oBranchPool = make_shared<Operation>("branchPool", module, (Sequential(branch_pool_1, branch_pool_2)).ptr(), false);

	addOperation(_oBranch1x1);
	addOperation(_oBranch7x7);
	addOperation(_oBranch7x7dbl);
	addOperation(_oBranchPool);
}