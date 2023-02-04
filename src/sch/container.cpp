# include <container.h>

# include <schd.h>
# include <ctx.h>
# include <operation.h>

# include <iostream>
# include <chrono>
# include <thread>
# include <torch/torch.h>

using namespace torch;
using namespace torch::nn;

using namespace FGPRS;

using namespace std;
using namespace std::chrono;

// void Container::analyze()
// {
// 	cout << "Yoohoo!\n";

// 	for (auto op : _operations)
// 	{
// 		// cout << "OK!\n";
// 		cout << op.analyze(5, 20, torch::randn({3, 1024, 1024}, kCUDA)) << endl;
// 		// cout << op.getName() << endl;
// 	}
// }