# include "str.hpp"
# include "ctx.hpp"

# include <cuda_runtime.h>

using namespace FGPRS;

MyStream::MyStream(MyContext* context, bool isFake) : context(context), _isFake(false)
{
	if (!_isFake)
		_stream = make_shared<CUDAStream>(getStreamFromPool(false, context->index));
}

bool MyStream::select()
{
	if (_isFake)
		return true;

	if (busy)
		return false;

	busy = true;
	_stream->select();
	setCurrentCUDAStream(*_stream);
	return true;
}

void MyStream::synchronize()
{
	if (!_isFake)
		_stream->synchronize();
}

void MyStream::release()
{
	if (!_isFake)
	{
		busy = false;
		synchronize();
	}
}

cudaStream_t MyStream::stream()
{
	return _stream->stream();
}