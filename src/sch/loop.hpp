# ifndef __LOOP__
# define __LOOP__

# include <cnt.hpp>

# include <memory>

# define PROPOSED_SCHEDULER	0
# define NOMPS_SCHEDULER		1
# define MP_SCHEDULERS			2
# define PMPS_SCHEDULER			3

namespace FGPRS
{
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

		void initialize(int deadlineContextIndex, Tensor dummyInput);
		void start(Tensor* input, int level = 0);
		void stop();
	};
}

# endif