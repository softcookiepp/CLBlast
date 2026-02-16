#include <clblast_vk.h>
#include <clblast_half.h>
#include <limits>
#include <cmath>

int main(int argc, char** argv)
{
#if 1
	float nan = std::numeric_limits<float>::signaling_NaN();
	float pInf = std::numeric_limits<float>::infinity();
	float nInf = -1.0*std::numeric_limits<float>::infinity();
	std::cout << "nan: " << HalfToFloat(FloatToHalf(nan)) << std::endl;
	std::cout << "pInf: " << HalfToFloat(FloatToHalf(pInf)) << std::endl;
	std::cout << "nInf: " << HalfToFloat(FloatToHalf(nInf)) << std::endl;
#else
	// here we test every possible 32-bit floating point value to see if any don't work.
	for (uint32_t i = 0; i < std::numeric_limits<uint32_t>::max(); i += 1)
	{
		float fp32Expected = *((float*)(&i));
		half fp16Result = FloatToHalf(fp32Expected);
		float fp32Result = HalfToFloat(fp16Result);
		if (std::abs(fp32Result - fp32Expected) > 1.0)
		{
			std::cout << "\nOriginal: " << fp32Expected << ", Converted: " << fp32Result;
		}
	}
	std::cout << std::endl;
#endif
}
