# pragma once

# include <cnt.hpp>

# include <memory>

namespace FGPRS
{
	enum SchedulerType { PROPOSED_SCHEDULER, NOMPS_SCHEDULER, MPS_SCHEDULER, PMPS_SCHEDULER, PMPSO_SCHEDULER };

	class Loop
	{
	private:
		shared_ptr<MyContainer> _container;
		double _frequency, _period;
		bool _stop = false;
		thread _th;
		string _name;
		int _index;

	public:
		int totalCount, compCount, missCount;

		Loop() {}
		Loop(shared_ptr<MyContainer> container);

		void start(int timer);
		void wait();
	};
}