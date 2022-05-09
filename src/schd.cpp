# include <schd.h>

using namespace FGPRS;

using namespace std;

bool Scheduler::initialize(int options[], int size)
{
	// cudaDeviceProp prop;
	// cudaGetDeviceProperties(&prop, 0);
	// maxSmCount = prop.multiProcessorCount;

	_poolSize = size;
	_pContextPool = (MyContext*)malloc(sizeof(MyContext) * _poolSize);
	bool result = true;

	for (int i; i < _poolSize; i++)
	{
		_pContextPool[i] = MyContext(options[i]);
		result &= _pContextPool[i].initialize();
	}

	return result;
}

bool Scheduler::initialize(int minSms, int maxSms)
{
	maxSms = maxSms <= maxSmCount ? maxSms : maxSmCount;
	_poolSize = maxSms - minSms + 1;
	_pContextPool = (MyContext*)malloc(sizeof(MyContext) * _poolSize);
	bool result = true;

	for (int i; i < _poolSize; i++)
	{
		_pContextPool[i] = MyContext(minSms++);
		result &= _pContextPool[i].initialize();
	}

	return result;
}

MyContext* Scheduler::selectContext(int smCount)
{
	MyContext* context = nullptr;

	for (int i; i < _poolSize; i++)
	{
		if (_pContextPool[i].smCount >= smCount && !_pContextPool[i].busy)
		{
			context = _pContextPool + i;
			break;
		}
	}

	if (context == nullptr)
	{
		for (int i = _poolSize - 1; i >= 0; i--)
		{
			if (!_pContextPool[i].busy)
			{
				context = _pContextPool + i;
				break;
			}
		}
	}

	return context;
}

bool Scheduler::selectDefaultContext()
{
	return MyContext::selectDefault();
}

bool Scheduler::releaseContext(MyContext context)
{
	return context.release();
}

bool Scheduler::destroyAll()
{
	bool result = true;

	for (int i; i < _poolSize; i++)
		result &= _pContextPool[i].destroy();
	
	free(_pContextPool);
	return true;
}