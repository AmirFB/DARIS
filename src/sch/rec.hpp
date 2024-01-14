#include <deque>
#include <chrono>
#include <vector>
#include <memory>

using namespace std;
using namespace chrono;

class OperationTrackingRecord
{
public:
	double executionTime;
	bool missed;
	int id;

	OperationTrackingRecord(int id) : id(id), missed(false) {}
};

class ModuleTrackingRecord
{
public:
	bool accepted, missed;
	double executionTime, responseTime;
	vector<shared_ptr<OperationTrackingRecord>> operations;

	ModuleTrackingRecord(bool accepted, int count) : accepted(accepted), operations(count), missed(false) {}

	void setResponseTime(double responseTime)
	{
		this->responseTime = responseTime;

		executionTime = 0;

		for (auto& operation : operations)
			executionTime += operation->executionTime;
	}
};