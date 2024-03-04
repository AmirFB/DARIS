# pragma once

# include "cnt.hpp"
# include "rec.hpp"

# include <vector>

using namespace std;

namespace FGPRS
{
	class ModuleTrackingRecord;
	class MyContainer;

	class ModuleTracker
	{
	private:
		MyContainer* _container;
		bool _first = true;

	public:
		static int windowSize;
		vector<shared_ptr<ModuleTrackingRecord>> records;
		vector<shared_ptr<ModuleTrackingRecord>> finalRecords;

		ModuleTracker() {}
		ModuleTracker(MyContainer* container) : _container(container) {}

		void addRecord(shared_ptr<ModuleTrackingRecord> record);
	};
}