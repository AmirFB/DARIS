# include <iostream>
# include <bitset>

# include <libsmctrl.h>

using namespace std;

int main()
{
	uint32_t numGpcs, numTpc;
	uint64_t* tpcForGpc;

	auto result = libsmctrl_get_gpc_info(&numGpcs, &tpcForGpc, 0);
	auto result2 = libsmctrl_get_tpc_info(&numTpc, 0);

	if (result != 0)
	{
		cout << "Error: " << result << endl;
		return 1;
	}

	if (result2 != 0)
	{
		cout << "Error: " << result2 << endl;
		return 1;
	}

	cout << "Number of GPCs: " << numGpcs << endl;

	for (uint32_t i = 0; i < numGpcs; i++)
	{
		cout << "GPC " << i << " in binary representation: " << bitset<sizeof(tpcForGpc[i]) * 8>(tpcForGpc[i]) << endl;
	}

	cout << "Number of TPCs: " << numTpc << endl;

	auto result3 = libsmctrl_get_tpc_info_cuda(&numTpc, 0);

	if (result3 != 0)
	{
		cout << "Error: " << result3 << endl;
		return 1;
	}

	cout << "Number of TPCs (CUDA): " << numTpc << endl;

	return 0;
}