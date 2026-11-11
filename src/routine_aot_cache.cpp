#include "routine_aot_cache.hpp"
#include "all-blas-kernels.hpp"

namespace clblast
{

using namespace tart::kernels;

// this will be used enough that it makes sense.
#ifndef VW_SWITCH
#define VW_SWITCH(NAMESPACE, KERNEL, VW, DEVICE) \
		switch(VW) \
		{ \
			case 1: return NAMESPACE::KERNEL##_vw1(DEVICE); \
			case 2: return NAMESPACE::KERNEL##_vw2(DEVICE); \
			case 4: return NAMESPACE::KERNEL##_vw4(DEVICE); \
			case 8: return NAMESPACE::KERNEL##_vw8(DEVICE); \
			case 16: return NAMESPACE::KERNEL##_vw16(DEVICE); \
			default: throw std::runtime_error("Invalid vector width"); \
		}
		
#endif

tart::kernel_ptr getAxpy(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth)
{
	if (dtype == tart::dtypes::float16)
	{
		VW_SWITCH(f16, axpy, vectorWidth, device);
	}
	else if(dtype == tart::dtypes::float32)
	{
		VW_SWITCH(f32, axpy, vectorWidth, device);
	}
	else if(dtype == tart::dtypes::float64)
	{
		VW_SWITCH(f64, axpy, vectorWidth, device);
	}
	else if(dtype == tart::dtypes::complex64)
	{
		VW_SWITCH(c64, axpy, vectorWidth, device);
	}
	else if(dtype == tart::dtypes::complex128)
	{
		VW_SWITCH(c128, axpy, vectorWidth, device);
	}
}

} // namespace clblast
