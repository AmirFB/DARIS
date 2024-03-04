# include "inception.hpp"

using namespace std;
using namespace FGPRS;
using namespace torch;
using namespace nn;

void InceptionDImpl::initialize(shared_ptr<MyContainer> module, string name, bool highPriority)
{
	moduleName = name;
	this->highPriority = highPriority;

	_oBranch3x3 = make_shared<Operation>("branch3x3", module, (Sequential(branch3x3_1, branch3x3_2)).ptr(), false);
	_oBranch7x7x3 = make_shared<Operation>("branch7x7x3", module,
		(Sequential(branch7x7x3_1, branch7x7x3_2, branch7x7x3_3, branch7x7x3_4)).ptr(), false);
	_oBranchPool = make_shared<Operation>("branchPool", module, (Sequential(branch_pool)).ptr(), false);

	addOperation(_oBranch3x3);
	addOperation(_oBranch7x7x3);
	addOperation(_oBranchPool);
}