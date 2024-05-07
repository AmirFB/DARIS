# pragma once

# include "cnt.hpp"

# include <c10/cuda/CUDAStream.h>

# include <cuda_runtime.h>

using namespace std;
using namespace c10::cuda;
using namespace at::cuda;

namespace FGPRS
{
	class MyContext;

	class MyStream
	{
	private:
		shared_ptr<CUDAStream> _stream;
		bool _isFake = false;

	public:
		bool busy = false;
		MyContext* context;

		MyStream() {}
		MyStream(MyContext* context, bool isFake = false);
		bool select();
		void synchronize();
		void release();
		cudaStream_t stream();
	};
}