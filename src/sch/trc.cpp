# include "rec.hpp"

using namespace FGPRS;

int ModuleTracker::windowSize;

void ModuleTracker::addRecord(shared_ptr<ModuleTrackingRecord> record)
{
	records.push_back(record);
}