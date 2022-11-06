# include <iostream>
# include <thread>
# include <pthread.h>
# include <chrono>
# include <string>

#include <torch/torch.h>

#include <torch/script.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>

# include <cuda.h>
# include <cudaTypedefs.h>
# include <cuda_runtime.h>

# include "ctx.h"
# include "schd.h"

# include "cif10.h"
# include "layer.h"
# include "resnet.h"

# include <cuda.h>
# include <cudaTypedefs.h>
# include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>

#define _UNIX03_THREADS 1
#include <limits.h>                                                            
#include <errno.h> 

using namespace std;
using namespace chrono;
using namespace torch;
using namespace FGPRS;

#define handle_error_en(en, msg) \
	do { errno = en; perror(msg); } while (0)
	// do { errno = en; perror(msg); exit(EXIT_FAILURE); } while (0)

# define NUM		2
# define REPEAT 5

CUcontext cu[NUM];

struct thread_data
{
	int index, sm;
};

struct thread_data thrd_d[NUM];

// auto layer = nn::Conv2d(nn::Conv2dOptions(16, 16, 11).stride(4).padding(2));
torch::nn::Sequential model = torch::nn::Sequential(
		torch::nn::Linear(3000, 50000));
auto input = torch::ones(3000, torch::kCUDA);
//torch::randn({16, 16, 11}, torch::kCUDA);

// void dummy(nn::Sequential model, Tensor input, unsigned sm)
void dummy(struct thread_data sm)
{
	static int round[NUM] = {0};
	// cout << this_thread::get_id() << ", " << sm.index << endl;
	cout << (Scheduler::selectContext((sm.sm + round[sm.index]) % 3 + 1)->select() ? "Selected" : "Oops!") << endl;
	// cuCtxSetCurrent(cu[sm.sm]);
	// printf("%d\n", cu[sm - 1]);
	// auto layer = nn::Conv2d(nn::Conv2dOptions(16, 16, 11).stride(4).padding(2));
	// auto model = nn::Sequential();
	// model->push_back(layer);
	// model->to(torch::kCUDA);
	// auto input = torch::randn({16, 16, 11}, torch::kCUDA);
	Tensor output;
	output = model->forward(input);

	// if (sm.sm == 1)
	// {
	// 	cout << "Syncing ...\n";
	// 	auto stream = at::cuda::getCurrentCUDAStream();
	// 	AT_CUDA_CHECK(cudaStreamSynchronize(stream));
	// 	cout << "Synced! (" << ++temp << ", " << sm.sm << ")\n";
	// }

	// torch::cuda::synchronize();
	stringstream ss;
	ss << "Th: " << sm.index << "    SM: " << ((sm.sm + round[sm.index]) % 3 + 1) << "    Rn: " << ++round[sm.index] << endl;
	string str = ss.str();
	cout << str;
	// sleep(1);
}

// void dummy(nn::Sequential model, Tensor input, unsigned sm)
void* dummy2(void* sm_)
{
	struct thread_data sm = *(struct thread_data*)sm_;
	dummy(sm);
	// torch::cuda::cudaStreamSynchronize();
	// pthread_exit(NULL);

	return NULL;
}

void* main_dummy(int index)
{
	pthread_t pth;

	for (int j = 0; j < REPEAT; j++)
	{
		thrd_d[index].index = index + 1;
		thrd_d[index].sm = index;
		pthread_create(&pth, NULL, dummy2, (void *)&thrd_d[index]);
		pthread_join(pth, NULL);
	}
}

void create_thread2(pthread_t *thr, unsigned &sm);

int main()
{
	bool result = Scheduler::initialize(1, 1);
	cout << (result ? "Scheduler initialized successfully." : "Scheduler initializatoin failed.") << endl;

	cout << fixed << setprecision(1)
		<< "Total Memory: " << Scheduler::getTotalMemoryGB()
		<< "GB\nFree Memory: " << Scheduler::getFreeMemoryGB()
		<< "GB\nMemory Percentage: " << Scheduler::getMemoryPercentage() << "%" << endl;
	
	model->to(torch::kCUDA);

	thread th[NUM];

	for (int i = 1; i <= NUM; i++)
	{
		
	}
	
	typedef high_resolution_clock Clock;

	for (int i = 0; i < NUM; i++)
	{
		th[i] = thread(main_dummy, i);
		cout << i << endl;
	}

	for (int i = 0; i < NUM; i++)
		th[i].join();
	
	return 0;

	// auto t1 = Clock::now();

	// for (int j = 0; j < REPEAT; j++)
	// {
	// 	for (int i = 1; i <= NUM; i++)
	// 	{
	// 		// thread th(dummy, model, input, i);
	// 		// dummy(model, input, i);;
	// 		// th[i - 1] = thread(dummy, i);
	// 		// th[i - 1].join();
	// 		// dummy(i);
	// 	}
	// }

	// auto t2 = Clock::now();
	// pth = (pthread_t *)malloc(sizeof(pthread_t) * NUM);

	// for (int j = 0; j < REPEAT; j++)
	// {
	// 	for (int i = 1; i <= NUM; i++)
	// 	{
	// 		// thread th(dummy, model, input, i);
	// 		// dummy(model, input, i);;
	// 		// th[i - 1] = thread(dummy, i);
	// 		// th[i - 1].join();
	// 		// dummy(i);
	// 		thrd_d[i - 1].index = i;
	// 		thrd_d[i - 1].sm = (i + j) % NUM;
	// 		pthread_create(&pth[i - 1], NULL, dummy2, (void *)&thrd_d[i - 1]);
	// 		// dummy2(&i);
	// 		// create_thread2(&pth[i - 1], i);
	// 		// pthread_join(pth[i - 1], NULL);
	// 	}

	// 	for (int i = 1; i <= NUM; i++)
	// 	{
	// 		// thread th(dummy, model, input, i);
	// 		// dummy(model, input, i);;
	// 		// th[i - 1] = thread(dummy, i);
	// 		// th[i - 1].join();
	// 		// dummy(i);
	// 		// pthread_create(&pth[i - 1], NULL, dummy2, (void*)&i);
	// 		pthread_join(pth[i - 1], NULL);
	// 	}

	// 	cout << j << "------------------------------------------" << j << endl;
	// }

	// sleep(1);
	// delete (pth);

	// auto t3 = Clock::now();

	// chrono::duration<double> d1, d2;
	// d1 = t2 - t1;
	// d2 = t3 - t2;

	// cout << d1.count() << endl;
	// cout << d2.count() << endl;
	// cout << (d2 - d1).count() / NUM / REPEAT * 1000000 << endl;
}

void create_thread2(pthread_t* thr, unsigned& sm)
{
	pthread_attr_t attr;
	pthread_attr_t* attrp;
	int s;

	attrp = NULL;

	size_t stack_size;
	void *sp;

	attrp = &attr;

	s = pthread_attr_init(&attr);

	if (s != 0)
		handle_error_en(s, "pthread_attr_init");

	s = pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	if (s != 0)
		handle_error_en(s, "pthread_attr_setdetachstate");

	s = pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);

	if (s != 0)
		handle_error_en(s, "pthread_attr_setinheritsched");

	stack_size = 1024 * 1024 * 1024;//strtoul(argv[1], NULL, 0);
	
	s = posix_memalign(&sp, sysconf(_SC_PAGESIZE), stack_size);

	if (s != 0)
		handle_error_en(s, "posix_memalign");

	// printf("posix_memalign() allocated at %p\n", sp);

	s = pthread_attr_setstack(&attr, sp, stack_size);

	if (s != 0)
		handle_error_en(s, "pthread_attr_setstack");

	int temp = sm;
	s = pthread_create(thr, attrp, &dummy2, (void*)&sm);
	// s = pthread_create(thr, NULL, dummy2, (void*)&sm);
	// cout << thr << endl;
	// cout << pthread_create(thr, NULL, dummy2, (void*)&sm) << endl;

	if (s != 0)
		handle_error_en(s, "pthread_create");

	if (attrp != NULL)
	{
		s = pthread_attr_destroy(attrp);

		if (s != 0)
			handle_error_en(s, "pthread_attr_destroy");
	}
}