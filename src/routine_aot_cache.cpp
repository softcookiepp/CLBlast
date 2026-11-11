#include "routine_aot_cache.hpp"
#include "all-blas-kernels.hpp"

namespace clblast
{

using namespace tart::kernels;

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

#ifndef DTYPE_VW_SWITCH
#define DTYPE_VW_SWITCH(KERNEL, DTYPE, VW, DEVICE) \
	if (DTYPE == tart::dtypes::float16) \
	{ \
		VW_SWITCH(f16, KERNEL, VW, DEVICE); \
	} \
	else if(DTYPE == tart::dtypes::float32) \
	{ \
		VW_SWITCH(f32, KERNEL, VW, DEVICE); \
	} \
	else if(DTYPE == tart::dtypes::float64) \
	{ \
		VW_SWITCH(f64, KERNEL, VW, DEVICE); \
	} \
	else if(DTYPE == tart::dtypes::complex64) \
	{ \
		VW_SWITCH(c64, KERNEL, VW, DEVICE); \
	} \
	else if(DTYPE == tart::dtypes::complex128) \
	{ \
		VW_SWITCH(c128, KERNEL, VW, DEVICE); \
	} \
	else \
	{ \
		throw std::runtime_error("dtype not implemented"); \
	}
	
#endif

#ifndef DTYPE_SWITCH
#define DTYPE_SWITCH(KERNEL, DTYPE, DEVICE) \
	if (DTYPE == tart::dtypes::float16) \
	{ \
		return f16::KERNEL(DEVICE); \
	} \
	else if(DTYPE == tart::dtypes::float32) \
	{ \
		return f32::KERNEL(DEVICE); \
	} \
	else if(DTYPE == tart::dtypes::float64) \
	{ \
		return f64::KERNEL(DEVICE); \
	} \
	else if(DTYPE == tart::dtypes::complex64) \
	{ \
		return c64::KERNEL(DEVICE); \
	} \
	else if(DTYPE == tart::dtypes::complex128) \
	{ \
		return c128::KERNEL(DEVICE); \
	} \
	else \
	{ \
		throw std::runtime_error("dtype not implemented"); \
		return nullptr; \
	}

#endif

tart::kernel_ptr getAmax(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(amax, dtype, device); }
tart::kernel_ptr getAmaxEpilogue(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(amax_epilogue, dtype, device); }

tart::kernel_ptr getAsum(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(asum, dtype, device); }
tart::kernel_ptr getAsumEpilogue(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(asum_epilogue, dtype, device); }

tart::kernel_ptr getCopy(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth) { DTYPE_VW_SWITCH(copy, dtype, vectorWidth, device); }
tart::kernel_ptr getCopyFast(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth) { DTYPE_VW_SWITCH(copy_fast, dtype, vectorWidth, device); }

tart::kernel_ptr getAxpy(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth)
{
	DTYPE_VW_SWITCH(axpy, dtype, vectorWidth, device);
}

tart::kernel_ptr getAxpyFaster(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth)
{
	DTYPE_VW_SWITCH(axpy_faster, dtype, vectorWidth, device);
}

tart::kernel_ptr getAxpyFastest(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth)
{
	DTYPE_VW_SWITCH(axpy_fastest, dtype, vectorWidth, device);
}

} // namespace clblast
