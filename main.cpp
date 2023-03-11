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

# include <schd.h>
# include <cnt.h>
# include <loop.h>

# include <cif10.h>
# include <cnt.h>
# include <resnet.h>

# include <tests.h>

# include <stdio.h>
# include <stdlib.h>
# include <unistd.h>

# include <c10/cuda/CUDACachingAllocator.h>

using namespace std;
using namespace std::chrono;
using namespace torch;
using namespace torch::nn;
using namespace FGPRS;

int main(int argc, char** argv)
{
	NoGradGuard no_grad;

	int smOptions[] = { 4, 10, 18, 36 };
	Scheduler::initialize(smOptions, 4);

	auto dummyData1 = torch::randn({ 1, 3, 224, 224 }, kCUDA);
	auto dummyData2 = torch::randn({ 1, 3, 2048, 2048 }, kCUDA);

	int level = 2;
	auto res = resnet18(1000);
	auto loop = Loop(res, 50);

	loop.initialize(level, dummyData1);
	this_thread::sleep_for(seconds(2));
	loop.start(&dummyData1, level);

	this_thread::sleep_for(milliseconds(200));
	loop.stop();
	// res->analyze(10, 50, dummyData2, 1);
	// res->analyze(10, 50, dummyData1);

	// cout << temp[0] << endl;
	// temp = res->forward(dummyData1);
	// cout << temp[0] << endl;

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