# include <iostream>
# include <fstream>
# include <iomanip>
# include <thread>
# include <pthread.h>
# include <chrono>
# include <string>
# include <cstdlib>
# include <future> 
# include <sys/stat.h>

# include <torch/torch.h>
# include <torch/script.h>
# include <c10/cuda/CUDAStream.h>
# include <ATen/cuda/CUDAContext.h>
# include <c10/cuda/CUDACachingAllocator.h>

# include <cuda.h>
# include <cudaTypedefs.h>
# include <cuda_runtime.h>

# include "ctx.h"
# include "schd.h"
# include "container.h"

# include "cif10.h"
# include "operation.h"
# include "resnet.h"

# include "tests.h"

# include <stdio.h>
# include <stdlib.h>
# include <unistd.h>

#include <c10/cuda/CUDACachingAllocator.h>

using namespace std;
using namespace std::chrono;
using namespace torch;
using namespace torch::nn;
using namespace FGPRS;

int main(int argc, char** argv)
{
	NoGradGuard no_grad;

	int smOptions[] = {4, 10, 18, 32, 36, 50, 58, 64};
	vector<int> realOptions{4, 10, 18, 36};
	Scheduler::initialize(smOptions, 8);
	// c10::cuda::CUDACachingAllocator::emptyCache();

	Tensor dummyData1 = torch::ones({1, 3, 224, 224}, kCUDA);
	Tensor dummyData2 = torch::ones({1, 3, 2048, 2048}, kCUDA);
	auto res = resnet158(1000);
	
	res->eval();
	res->to(kCUDA);

	Scheduler::selectDefaultContext();
	// auto dummy = res->forward(dummyData1);
	auto temp = res->analyze(dummyData1, realOptions);
	cout << endl << endl << endl << endl << endl << endl << endl << endl << endl;
	temp = res->analyze(dummyData1, realOptions);
	cout << endl << endl << endl;
	temp = res->analyze(dummyData2, realOptions);

	// char *op = argv[1];
	// mkdir("results", 0777 );

	// if (!strcmp(op, "clear"))
	// {
	// 	cout << "Removing previous results of \"" << argv[2] << "\" simulation\n";
	// 	remove((string("results/") + string(argv[2]) + ".csv").c_str());
	// }

	// else if (!strcmp(op, "speedup"))
	// 	testSpeedup(&argv[2]);

	// // else if (!strcmp(op, "concurrency"))
	// // 	testConcurrency(&argv[2]);

	// else if (!strcmp(op, "tailing"))
	// 	testTailing(&argv[2]);

	// else if (!strcmp(op, "interference"))
	// 	testInterference(&argv[2]);
}