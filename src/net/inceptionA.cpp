# include "inception.hpp"

# include <ctx.hpp>

# include <future>

using namespace std;
using namespace FGPRS;
using namespace torch;
using namespace nn;

Tensor InceptionA::forwardNL(Tensor input, MyContext* ctx, MyStream* mainStream)
{
	future<Tensor> th1x1 = async(launch::async, [&](MyStream* str)
		{
			str->context->select();
			str->select();
			Tensor branch1x1 = this->branch1x1->forward(input);
			str->release();
			return branch1x1;
		}, ctx->getSecondaryStream(mainStream));

	future<Tensor> th5x5 = async(launch::async, [&](MyStream* str)
		{
			str->context->select();
			str->select();
			Tensor branch5x5 = this->branch5x5_1->forward(input);
			branch5x5 = this->branch5x5_2->forward(branch5x5);
			str->release();
			return branch5x5;
		}, ctx->getSecondaryStream(mainStream));

	future<Tensor> th3x3dbl = async(launch::async, [&](MyStream* str)
		{
			str->context->select();
			str->select();
			Tensor branch3x3dbl = this->branch3x3dbl_1->forward(input);
			branch3x3dbl = this->branch3x3dbl_2->forward(branch3x3dbl);
			branch3x3dbl = this->branch3x3dbl_3->forward(branch3x3dbl);
			str->release();
			return branch3x3dbl;
		}, ctx->getSecondaryStream(mainStream));

	future<Tensor> thPool = async(launch::async, [&](MyStream* str)
		{
			str->context->select();
			str->select();
			Tensor branch_pool = this->branch_pool_1->forward(input);
			branch_pool = this->branch_pool_2->forward(branch_pool);
			str->release();
			return branch_pool;
		}, ctx->getSecondaryStream(mainStream));

	return cat({ th1x1.get(), th5x5.get(), th3x3dbl.get(), thPool.get() }, 1);
}