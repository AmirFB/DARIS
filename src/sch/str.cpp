# include "str.hpp"
# include "ctx.hpp"

# include <cuda_runtime.h>

using namespace FGPRS;

MyStream::MyStream(MyContext* context) : _context(context)
{
	_stream = make_shared<CUDAStream>(getStreamFromPool(false, context->index));
}

bool MyStream::select()
{
	if (busy)
		return false;

	busy = true;
	_stream->select();
	setCurrentCUDAStream(*_stream);
	return true;
}

void MyStream::synchronize()
{
	_stream->synchronize();
}

void MyStream::release()
{
	busy = false;
	synchronize();
}