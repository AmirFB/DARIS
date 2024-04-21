# include "inception.hpp"

using namespace std;
using namespace FGPRS;
using namespace torch;
using namespace nn;

Tensor InceptionC::forwardNL(Tensor input, MyContext* ctx, MyStream* mainStream)
{
	// future<Tensor> th1x1 = async(launch::async, [&](MyStream* str)
	// 	{
	// 		str->context->select();
	// 		str->select();
	Tensor branch1x1 = this->branch1x1->forward(input);
	// 	str->release();
	// 	return branch1x1;
	// }, ctx->getSecondaryStream(mainStream));

	future<Tensor> th7x7 = async(launch::async, [&](MyStream* str)
		{
			str->context->select();
			str->select();
			Tensor branch7x7 = this->branch7x7_1->forward(input);
			branch7x7 = this->branch7x7_2->forward(branch7x7);
			branch7x7 = this->branch7x7_3->forward(branch7x7);
			str->release();
			return branch7x7;
		}, ctx->getSecondaryStream(mainStream));

	future<Tensor> th7x7dbl = async(launch::async, [&](MyStream* str)
		{
			str->context->select();
			str->select();
			Tensor branch7x7dbl = this->branch7x7dbl_1->forward(input);
			branch7x7dbl = this->branch7x7dbl_2->forward(branch7x7dbl);
			branch7x7dbl = this->branch7x7dbl_3->forward(branch7x7dbl);
			branch7x7dbl = this->branch7x7dbl_4->forward(branch7x7dbl);
			branch7x7dbl = this->branch7x7dbl_5->forward(branch7x7dbl);
			str->release();
			return branch7x7dbl;
		}, ctx->getSecondaryStream(mainStream));

	// future<Tensor> thPool = async(launch::async, [&](MyStream* str)
	// 	{
	// 		str->context->select();
	// 		str->select();
	Tensor branch_pool = this->branch_pool_1->forward(input);
	branch_pool = this->branch_pool_2->forward(branch_pool);
	// 	str->release();
	// 	return branch_pool;
	// }, ctx->getSecondaryStream(mainStream));

	return cat({ branch1x1, th7x7.get(), th7x7dbl.get(), branch_pool }, 1);
}