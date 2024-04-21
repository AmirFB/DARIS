# include "resnet.hpp"
# include "unet.hpp"
# include "schd.hpp"
# include "cnt.hpp"

using namespace std;
using namespace torch;
using namespace torch::nn;
using namespace FGPRS;

const int resnetMaxFps = 650, unetMaxFps = 200;

Scenario scenario1()
{
	// auto scenario = Scenario()
}