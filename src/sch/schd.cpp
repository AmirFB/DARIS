# include <schd.h>

# include <iostream>

# include <cuda.h>
# include <cudaTypedefs.h>
# include <cuda_runtime.h>

using namespace FGPRS;

using namespace std;

int Scheduler::maxSmCount;
int *Scheduler::smOptions;
int Scheduler::poolSize;
MyContext *Scheduler::_pContextPool;
MyContext *Scheduler::_defaultContext;

bool Scheduler::initialize(int options[], int size)
{
	bool result = true;
	cudaDeviceProp prop;

	cudaGetDeviceProperties(&prop, 0);
	maxSmCount = prop.multiProcessorCount;

	poolSize = size + 1;
	_pContextPool = (MyContext*)malloc(sizeof(MyContext) * poolSize);

	free(smOptions);
	smOptions = (int *)malloc(sizeof(int) * poolSize);

	for (int i = 0; i < (poolSize - 1); i++)
	{
		smOptions[i] = max(options[i], 1);
		_pContextPool[i] = MyContext(smOptions[i]);
		result &= _pContextPool[i].initialize();
	}

	smOptions[poolSize - 1] = maxSmCount;
	_pContextPool[poolSize - 1] = MyContext();

	return result;
}

bool Scheduler::initialize(int minSms, int maxSms, int step)
{
	/* Depricated for now! */

	// bool result = true;
	// cudaDeviceProp prop;

	// cudaGetDeviceProperties(&prop, 0);
	// maxSmCount = prop.multiProcessorCount;

	// maxSms = maxSms <= maxSmCount ? maxSms : maxSmCount;
	// poolSize = (maxSms - minSms) / step + 1;
	// _pContextPool = (MyContext*)malloc(sizeof(MyContext) * poolSize);

	// free(smOptions);
	// smOptions = (int *)malloc(sizeof(int) * poolSize);

	// for (int i = 0; i < poolSize; i += 2)
	// {
	// 	smOptions[i] = minSms + i;
	// 	_pContextPool[i] = MyContext(minSms + i);
	// 	result &= _pContextPool[i].initialize();
	// }

	// return result;

	return false;
}

MyContext* Scheduler::selectContext(int smCount)
{
	for (int i = 0; i < poolSize; i++)
	{
		// cout << "Here: " << _pContextPool[i].smCount << ", " << _pContextPool[i].busy << endl;
		if (_pContextPool[i].smCount >= smCount && !_pContextPool[i].busy)
		{
			// cout << "Going with: " << _pContextPool[i].smCount << " SMs.\n";
			return _pContextPool + i;
		}
	}

	for (int i = poolSize - 1; i >= 0; i--)
	{
		if (!_pContextPool[i].busy)
		{
			// cout << "Going with: " << _pContextPool[i].smCount << " SMs.\n";
			return _pContextPool + i;
		}
	}
	
	cout << "Going NULL: " << smCount << ", " << poolSize << endl;

	for (int i = 0; i < poolSize; i++)
		cout << "\t" << i << ": " << _pContextPool[i].smCount << ", " << _pContextPool[i].busy << endl;

	return _defaultContext;
}

MyContext* Scheduler::selectContextByIndex(int index)
{
	return _pContextPool + index;
}

MyContext* Scheduler::selectContextByQueueLoad(double* executionTime)
{
	// cout << "a: " << _pContextPool[0].queueDuration << endl;
	// cout << "b: " << _pContextPool[0].smCount << endl;
	// cout << "c: " << executionTime[_pContextPool[0].smCount] << endl << endl;

	double min = _pContextPool[0].queueDuration + executionTime[_pContextPool[0].smCount] * 1000000;
	int index = 0;

	for (int i = 1; i < poolSize; i++)
	{
		if ((_pContextPool[i].queueDuration + executionTime[_pContextPool[i].smCount] * 1000000) < min)
		{
			index = i;
			min = _pContextPool[i].queueDuration + executionTime[_pContextPool[i].smCount] * 1000000;
		}
	}

	return _pContextPool + index;
}

bool Scheduler::selectDefaultContext()
{
	return MyContext::selectDefault();
}

bool Scheduler::releaseContext(MyContext context)
{
	return context.release(0);
}

bool Scheduler::destroyAll()
{
	bool result = true;

	for (int i = 0; i < (poolSize - 1); i++)
		result &= _pContextPool[i].destroy();
	
	free(_pContextPool);
	return true;
}

float Scheduler::getTotalMemoryMB()
{
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	return total / 1024. / 1024.;
}

float Scheduler::getTotalMemoryGB()
{
	return Scheduler::getTotalMemoryMB() / 1024;
}

float Scheduler::getFreeMemoryMB()
{
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	return free / 1024. / 1024.;
}

float Scheduler::getFreeMemoryGB()
{
	return Scheduler::getFreeMemoryMB() / 1024;
}

float Scheduler::getMemoryPercentage()
{
	return Scheduler::getFreeMemoryMB() / Scheduler::getTotalMemoryMB() * 100;
}