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
		MyContext* _context;

	public:
		bool busy = false;

		MyStream(MyContext* context);
		bool select();
		void synchronize();
		void release();
	};
}