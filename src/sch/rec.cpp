# include "rec.hpp"

using namespace FGPRS;
using namespace std;

OperationTrackingRecord::OperationTrackingRecord(Operation* operation, bool missed, int executionTime) :
	operation(operation), missed(missed), executionTime(executionTime)
{
}

ModuleTrackingRecord::ModuleTrackingRecord(MyContainer* container, bool accepted) :
	container(container), accepted(accepted), missed(false)
{
	if (accepted)
		operations = vector<shared_ptr<OperationTrackingRecord>>(container->operations.size());
}

void ModuleTrackingRecord::setOperationExecutionTime(
	Operation* operation, bool missed, int executionTime)
{
	operations[operation->id] = make_shared<OperationTrackingRecord>(operation, missed, executionTime);
}

void ModuleTrackingRecord::setOperationWret(Operation* operation, int wret)
{
	operations[operation->id]->wret = wret;
}

void ModuleTrackingRecord::setResponseTime(int responseTime)
{
	this->responseTime = responseTime;

	executionTime = 0;
	wret = 0;

	for (auto& operation : operations)
	{
		executionTime += operation->executionTime;
		wret += operation->wret;
	}
}