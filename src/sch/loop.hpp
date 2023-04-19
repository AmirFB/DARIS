# ifndef __LOOP__
# define __LOOP__

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
		Loop() {}
		Loop(string name, shared_ptr<MyContainer> container, double _frequency, int index = -1);

		void initialize(int deadlineContextIndex, Tensor dummyInput, SchedulerType type, int level);
		void start(Tensor* input, SchedulerType type, int level = 0);
		void stop();
		void wait();
	};
}

# endif