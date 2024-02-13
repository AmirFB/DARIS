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

	public:
		static int windowSize;
		vector<shared_ptr<ModuleTrackingRecord>> records;

		ModuleTracker() {}
		ModuleTracker(MyContainer* container) : _container(container) {}

		void addRecord(shared_ptr<ModuleTrackingRecord> record);
	};
}