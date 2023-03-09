# ifndef __LOOP__
# define __LOOP__

# include <cnt.h>

# include <memory>

namespace FGPRS
{
	class Loop
	{
	private:
		shared_ptr<MyContainer> _container;
		double _frequency, _period;
		bool _stop = false;
		thread _th;

	public:
		Loop(shared_ptr<MyContainer> container, double _frequency);
		// Loop(shared_ptr<MyContainer> container, double _period);

		void start(Tensor* input);
		void stop();
	};
}

# endif