# include "rec.hpp"

using namespace FGPRS;

int ModuleTracker::windowSize;

void ModuleTracker::addRecord(shared_ptr<ModuleTrackingRecord> record)
{
	records.push_back(record);

	// if (!_first)
	if (records.size() >= _container->frequency)
		finalRecords.push_back(record);

	_first = false;
}