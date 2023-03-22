# ifndef CTXD_H
# define CTXD_H

// # include <ctx.hpp>

# include <cstdint>
# include <mutex>
# include <iostream>
# include <memory>

using namespace std;

namespace FGPRS
{
	class MyContext;

	class ContextData
	{
	public:
		MyContext* context;
		double isolatedExecutionTime, occupiedExecutionTime;
		int smCount;

		ContextData(MyContext* context);
		ContextData(MyContext* context, double isolatedExecutionTime, double occupiedExecutionTime);
		virtual void stackExecutionTime(ContextData ctxData);
	};
}

# endif