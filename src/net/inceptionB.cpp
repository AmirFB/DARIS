# include "inception.hpp"

using namespace std;
using namespace FGPRS;
using namespace torch;
using namespace nn;

Tensor InceptionB::forwardNL(Tensor input, MyContext* ctx, MyStream* mainStream)
{
	// future th3x3 = async(launch::async, [&](MyStream* str)
	// 	{
	// 		str->context->select();
	// 		str->select();
	Tensor branch3x3 = this->branch3x3->forward(input);
	// 	str->release();
	// 	return branch3x3;
	// }, ctx->getSecondaryStream(mainStream));

	future th3x3dbl = async(launch::async, [&](MyStream* str)
		{
			str->context->select();
			str->select();
			Tensor branch3x3dbl = this->branch3x3dbl_1->forward(input);
			branch3x3dbl = this->branch3x3dbl_2->forward(branch3x3dbl);
			branch3x3dbl = this->branch3x3dbl_3->forward(branch3x3dbl);
			str->release();
			return branch3x3dbl;
		}, ctx->getSecondaryStream(mainStream));

	// future thpool = async(launch::async, [&](MyStream* str)
	// 	{
	// 		str->context->select();
	// 		str->select();
	Tensor branch_pool = this->branch_pool->forward(input);
	// 	str->release();
	// 	return branch_pool;
	// }, ctx->getSecondaryStream(mainStream));

	return cat({ branch3x3, th3x3dbl.get(), branch_pool }, 1);
}