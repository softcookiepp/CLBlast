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

#ifndef DUAL_VW_SWITCH
#define DUAL_VW_SWITCH(NAMESPACE, KERNEL, VW1, VW2, DEVICE) \
		switch(VW1) \
		{ \
			case 1: { VW_SWITCH(NAMESPACE, KERNEL##_vw1, VW2, DEVICE) } \
			case 2: { VW_SWITCH(NAMESPACE, KERNEL##_vw2, VW2, DEVICE) } \
			case 4: { VW_SWITCH(NAMESPACE, KERNEL##_vw4, VW2, DEVICE) } \
			case 8: { VW_SWITCH(NAMESPACE, KERNEL##_vw8, VW2, DEVICE) } \
			case 16: { VW_SWITCH(NAMESPACE, KERNEL##_vw16, VW2, DEVICE) } \
			default: throw std::runtime_error("Invalid vector width"); \
		}
#endif

#ifndef DTYPE_DUAL_VW_SWITCH
#define DTYPE_DUAL_VW_SWITCH(DTYPE, KERNEL, VW1, VW2, DEVICE) \
	if (DTYPE == tart::dtypes::float16) \
	{ \
		DUAL_VW_SWITCH(f16, KERNEL, VW1, VW2, DEVICE); \
	} \
	else if(DTYPE == tart::dtypes::float32) \
	{ \
		DUAL_VW_SWITCH(f32, KERNEL, VW1, VW2, DEVICE); \
	} \
	else if(DTYPE == tart::dtypes::float64) \
	{ \
		DUAL_VW_SWITCH(f64, KERNEL, VW1, VW2, DEVICE); \
	} \
	else if(DTYPE == tart::dtypes::complex64) \
	{ \
		DUAL_VW_SWITCH(c64, KERNEL, VW1, VW2, DEVICE); \
	} \
	else if(DTYPE == tart::dtypes::complex128) \
	{ \
		DUAL_VW_SWITCH(c128, KERNEL, VW1, VW2, DEVICE); \
	} \
	else \
	{ \
		throw std::runtime_error("dtype not implemented"); \
		return nullptr; \
	}
	
#endif

#ifndef DUAL_VW_SUBGROUP_SWITCH
#define DUAL_VW_SUBGROUP_SWITCH(NAMESPACE, KERNEL, SUBGROUP_SUPPORT, VW1, VW2, DEVICE) \
	if (SUBGROUP_SUPPORT) { DUAL_VW_SWITCH(NAMESPACE, KERNEL##_##sub1, VW1, VW2, DEVICE) } \
	else { DUAL_VW_SWITCH(NAMESPACE, KERNEL##_##sub0, VW1, VW2, DEVICE) } \
	
#endif

#ifndef SUBGROUP_SWITCH_VW1_VW1
#define SUBGROUP_SWITCH_VW1_VW1(NAMESPACE, KERNEL, SUBGROUP_SUPPORT, DEVICE) \
	if (SUBGROUP_SUPPORT) { return NAMESPACE::KERNEL##_sub1##_vw1_vw1(DEVICE); } \
	else { return NAMESPACE::KERNEL##_sub1##_vw1_vw1(DEVICE); } \
	
#endif

#ifndef DTYPE_DUAL_VW_SUBGROUP_SWITCH
#define DTYPE_DUAL_VW_SUBGROUP_SWITCH(DTYPE, KERNEL, SUBGROUP_SUPPORT, DEVICE) \
	if (DTYPE == tart::dtypes::float16) \
	{ \
		SUBGROUP_SWITCH_VW1_VW1(f16, KERNEL, SUBGROUP_SUPPORT, DEVICE); \
	} \
	else if(DTYPE == tart::dtypes::float32) \
	{ \
		SUBGROUP_SWITCH_VW1_VW1(f32, KERNEL, SUBGROUP_SUPPORT, DEVICE); \
	} \
	else if(DTYPE == tart::dtypes::float64) \
	{ \
		SUBGROUP_SWITCH_VW1_VW1(f64, KERNEL, SUBGROUP_SUPPORT, DEVICE); \
	} \
	else if(DTYPE == tart::dtypes::complex64) \
	{ \
		SUBGROUP_SWITCH_VW1_VW1(c64, KERNEL, SUBGROUP_SUPPORT, DEVICE); \
	} \
	else if(DTYPE == tart::dtypes::complex128) \
	{ \
		SUBGROUP_SWITCH_VW1_VW1(c128, KERNEL, SUBGROUP_SUPPORT, DEVICE); \
	} \
	else \
	{ \
		throw std::runtime_error("dtype not implemented"); \
		return nullptr; \
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

tart::kernel_ptr getDot(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(dot, dtype, device); }
tart::kernel_ptr getDotEpilogue(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(dot_epilogue, dtype, device); }

tart::kernel_ptr getNrm2(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(nrm2, dtype, device); }
tart::kernel_ptr getNrm2Epilogue(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(nrm2_epilogue, dtype, device); }

tart::kernel_ptr getCopy(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth) { DTYPE_VW_SWITCH(copy, dtype, vectorWidth, device); }
tart::kernel_ptr getCopyFast(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth) { DTYPE_VW_SWITCH(copy_fast, dtype, vectorWidth, device); }

tart::kernel_ptr getScal(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth) { DTYPE_VW_SWITCH(scal, dtype, vectorWidth, device); }
tart::kernel_ptr getScalFast(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth) { DTYPE_VW_SWITCH(scal_fast, dtype, vectorWidth, device); }

tart::kernel_ptr getSwap(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth) { DTYPE_VW_SWITCH(swap, dtype, vectorWidth, device); }
tart::kernel_ptr getSwapFast(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth) { DTYPE_VW_SWITCH(swap_fast, dtype, vectorWidth, device); }

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
tart::kernel_ptr getAxpyBatched(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(axpy_batched, dtype, device); }

tart::kernel_ptr getHad(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth) { DTYPE_VW_SWITCH(had, dtype, vectorWidth, device); }
tart::kernel_ptr getHadFaster(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth) { DTYPE_VW_SWITCH(had_faster, dtype, vectorWidth, device); }
tart::kernel_ptr getHadFastest(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth) { DTYPE_VW_SWITCH(had_fastest, dtype, vectorWidth, device); }

tart::kernel_ptr getGemv(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(gemv, dtype, device); }
tart::kernel_ptr getGemvFast(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth) { DTYPE_VW_SWITCH(gemv_fast, dtype, vectorWidth, device); }
tart::kernel_ptr getGemvFastRot(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth) { DTYPE_VW_SWITCH(gemv_fast_rot, dtype, vectorWidth, device); }

tart::kernel_ptr getGer(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(ger, dtype, device); }
tart::kernel_ptr getHer(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(her, dtype, device); }
tart::kernel_ptr getHer2(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(her2, dtype, device); }
tart::kernel_ptr getTrsvForward(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(trsv_forward, dtype, device); }
tart::kernel_ptr getTrsvBackward(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(trsv_backward, dtype, device); }
tart::kernel_ptr getTrsvFillVector(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(trsv_fill_vector, dtype, device); }

tart::kernel_ptr getGemm(const tart::device_ptr& device, const tart::DType& dtype, const bool subgroupSupport, const uint32_t vwm, const uint32_t vwn)
{
	DTYPE_DUAL_VW_SUBGROUP_SWITCH(dtype, gemm, subgroupSupport, device)
}
tart::kernel_ptr getGemmUpper(const tart::device_ptr& device, const tart::DType& dtype, const bool subgroupSupport, const uint32_t vwm, const uint32_t vwn)
{
	DTYPE_DUAL_VW_SUBGROUP_SWITCH(dtype, gemm_upper, subgroupSupport, device)
}
tart::kernel_ptr getGemmLower(const tart::device_ptr& device, const tart::DType& dtype, const bool subgroupSupport, const uint32_t vwm, const uint32_t vwn)
{
	DTYPE_DUAL_VW_SUBGROUP_SWITCH(dtype, gemm_lower, subgroupSupport, device)
}
tart::kernel_ptr getGemmBatched(const tart::device_ptr& device, const tart::DType& dtype, const bool subgroupSupport, const uint32_t vwm, const uint32_t vwn)
{
	DTYPE_DUAL_VW_SUBGROUP_SWITCH(dtype, gemm_batched, subgroupSupport, device)
}
tart::kernel_ptr getGemmStridedBatched(const tart::device_ptr& device, const tart::DType& dtype, const bool subgroupSupport, const uint32_t vwm, const uint32_t vwn)
{
	DTYPE_DUAL_VW_SUBGROUP_SWITCH(dtype, gemm_strided_batched, subgroupSupport, device)
}

tart::kernel_ptr getGemmDirect(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vwm, const uint32_t vwn)
{
	DTYPE_DUAL_VW_SWITCH(dtype, gemm_direct, vwm, vwn, device)
}
tart::kernel_ptr getGemmDirectBatched(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vwm, const uint32_t vwn)
{
	DTYPE_DUAL_VW_SWITCH(dtype, gemm_direct_batched, vwm, vwn, device)
}
tart::kernel_ptr getGemmDirectStridedBatched(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vwm, const uint32_t vwn)
{
	DTYPE_DUAL_VW_SWITCH(dtype, gemm_direct_strided_batched, vwm, vwn, device)
}

tart::kernel_ptr getConvertHermitianLower(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(convert_hermitian_lower, dtype, device); }
tart::kernel_ptr getConvertHermitianUpper(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(convert_hermitian_upper, dtype, device); }
tart::kernel_ptr getConvertSymmetricLower(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(convert_symmetric_lower, dtype, device); }
tart::kernel_ptr getConvertSymmetricUpper(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(convert_symmetric_upper, dtype, device); }
tart::kernel_ptr getConvertTriangularLower(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(convert_triangular_lower, dtype, device); }
tart::kernel_ptr getConvertTriangularUpper(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(convert_triangular_upper, dtype, device); }
tart::kernel_ptr getFillMatrix(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(level3_fill_matrix, dtype, device); }
tart::kernel_ptr getTransposeMatrixFast(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth) { DTYPE_VW_SWITCH(transpose_fast, dtype, vectorWidth, device); }
tart::kernel_ptr getTransposePadMatrix(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(transpose_pad_matrix, dtype, device); }
tart::kernel_ptr getTransposeMatrix(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(transpose_matrix, dtype, device); }
tart::kernel_ptr getCopyMatrixFast(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(copy_matrix_fast, dtype, device); }
tart::kernel_ptr getCopyPadMatrix(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(copy_pad_matrix, dtype, device); }
tart::kernel_ptr getCopyMatrix(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(copy_matrix, dtype, device); }

tart::kernel_ptr getTransposePadMatrixBatched(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(transpose_pad_matrix_batched, dtype, device); }
tart::kernel_ptr getTransposeMatrixBatched(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(transpose_matrix_batched, dtype, device); }
tart::kernel_ptr getCopyPadMatrixBatched(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(copy_pad_matrix_batched, dtype, device); }
tart::kernel_ptr getCopyMatrixBatched(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(copy_matrix_batched, dtype, device); }

tart::kernel_ptr getTransposePadMatrixStridedBatched(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(transpose_pad_matrix_strided_batched, dtype, device); }
tart::kernel_ptr getTransposeMatrixStridedBatched(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(transpose_matrix_strided_batched, dtype, device); }
tart::kernel_ptr getCopyPadMatrixStridedBatched(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(copy_pad_matrix_strided_batched, dtype, device); }
tart::kernel_ptr getCopyMatrixStridedBatched(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(copy_matrix_strided_batched, dtype, device); }

tart::kernel_ptr getConvgemm(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vwm, const uint32_t vwn)
{
	DTYPE_DUAL_VW_SWITCH(dtype, convgemm_normal, vwm, vwn, device)
}
tart::kernel_ptr getConvgemmWithIm2col(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vwm, const uint32_t vwn)
{
	DTYPE_DUAL_VW_SWITCH(dtype, convgemm_kernel, vwm, vwn, device)
}
tart::kernel_ptr getCol2im(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(col2im_kernel_normal, dtype, device); }
tart::kernel_ptr getIm2col(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(im2col_kernel_normal, dtype, device); }

} // namespace clblast
