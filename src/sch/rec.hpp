# pragma once

# include "opr.hpp"
# include "cnt.hpp"

# include <deque>
# include <chrono>
# include <vector>
# include <memory>

using namespace std;
using namespace chrono;

namespace FGPRS
{
	class Operation;
	class MyContainer;

	class OperationTrackingRecord
	{
	public:
		shared_ptr<Operation> operation;
		int executionTime, wret;
		bool missed;

		OperationTrackingRecord() {}
		OperationTrackingRecord(Operation* operation, bool missed, int executionTime);
	};

	class ModuleTrackingRecord
	{
	public:
		bool accepted, missed;
		int executionTime, responseTime, wret;
		MyContainer* container;
		vector<shared_ptr<OperationTrackingRecord>> operations;

		ModuleTrackingRecord(MyContainer* container, bool accepted);
		void setOperationExecutionTime(Operation* operation, bool missed, int executionTime);
		void setOperationWret(Operation* operation, int wret);

		void setResponseTime(int responseTime);
	};
}