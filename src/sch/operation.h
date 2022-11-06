# include "ctx.h"
# include "schd.h"

# include <torch/torch.h>
# include <torch/script.h>
# include <c10/cuda/CUDAStream.h>
# include <ATen/cuda/CUDAContext.h>

using namespace std;
using namespace torch;
using namespace torch::nn;
using namespace FGPRS;

# define MAX_DATA_SIZE	20

struct operationData
{
	bool used;
	AnyModule module;
	Tensor *input;
	Tensor *output;
	int smCount, realSMs;
	MyContext *context;
};

typedef struct operationData OperationData;

// OperationData* buffer[MAX_DATA_SIZE] = { 0 };

// OperationData* selectData()
// {
// 	static int index = 0;
// 	int initialIndex = index;

// 	do
// 	{
// 		if (buffer[index] == NULL)
// 		{
// 			buffer[index] = (OperationData *)malloc(sizeof(OperationData));
// 			break;
// 		}

// 		while (buffer[index]->used)
// 		{
// 			index++;
// 			index = index < MAX_DATA_SIZE ? index : 0;
// 			break;
// 		}
// 	} while (true);

// 	return buffer[index++];
// }

// void* ptRoutine(void* pars)
// {
// 	// OperationData *data = (OperationData*)pars;
// 	// auto context = Scheduler::selectContext(data->smCount);
// 	// data->realSMs = context->smCount;
// 	// context->select();
// 	// *data->output = (data->module)->forward(*data->input);
// 	// context->release();
// }

// int execute(AnyModule* module, Tensor* input, Tensor* output, int smCount)
// {
// 	// OperationData data = {module, input, output, smCount};
// 	// pthread_t pth;
// 	// pthread_create(&pth, NULL, ptRoutine, (void *)&data);
// 	// pthread_join(pth, NULL);
// 	// ptRoutine((void *)&data);

// 	// return *data->tensor;
// }

// void join(int index)
// {

// }

namespace FGPRS
{
	class Operation
	{
		private:
		AnyModule *_module;
		Sequential *_sequence;
		shared_ptr<Tensor> _output;
		double *_execTime;
		bool _isSequential;
		double _relativeDeadline;
		double _absoulteDeadline;

		public:
		Operation(AnyModule *module);
		Operation(Sequential *sequence);
		Tensor *forward(Tensor *input);
		void initialize(Tensor *dummyInput, int warmup, int repeat);
	};
}