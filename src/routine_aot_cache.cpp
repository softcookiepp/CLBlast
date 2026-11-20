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
		VW_SWITCH(blas, F16##KERNEL, VW, DEVICE); \
	} \
	else if(DTYPE == tart::dtypes::float32) \
	{ \
		VW_SWITCH(blas, F32##KERNEL, VW, DEVICE); \
	} \
	else if(DTYPE == tart::dtypes::float64) \
	{ \
		VW_SWITCH(blas, F64##KERNEL, VW, DEVICE); \
	} \
	else if(DTYPE == tart::dtypes::complex64) \
	{ \
		VW_SWITCH(blas, C64##KERNEL, VW, DEVICE); \
	} \
	else if(DTYPE == tart::dtypes::complex128) \
	{ \
		VW_SWITCH(blas, C128##KERNEL, VW, DEVICE); \
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
		DUAL_VW_SWITCH(blas, F16##KERNEL, VW1, VW2, DEVICE); \
	} \
	else if(DTYPE == tart::dtypes::float32) \
	{ \
		DUAL_VW_SWITCH(blas, F32##KERNEL, VW1, VW2, DEVICE); \
	} \
	else if(DTYPE == tart::dtypes::float64) \
	{ \
		DUAL_VW_SWITCH(blas, F64##KERNEL, VW1, VW2, DEVICE); \
	} \
	else if(DTYPE == tart::dtypes::complex64) \
	{ \
		DUAL_VW_SWITCH(blas, C64##KERNEL, VW1, VW2, DEVICE); \
	} \
	else if(DTYPE == tart::dtypes::complex128) \
	{ \
		DUAL_VW_SWITCH(blas, C128##KERNEL, VW1, VW2, DEVICE); \
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
		SUBGROUP_SWITCH_VW1_VW1(blas, F16##KERNEL, SUBGROUP_SUPPORT, DEVICE); \
	} \
	else if(DTYPE == tart::dtypes::float32) \
	{ \
		SUBGROUP_SWITCH_VW1_VW1(blas, F32##KERNEL, SUBGROUP_SUPPORT, DEVICE); \
	} \
	else if(DTYPE == tart::dtypes::float64) \
	{ \
		SUBGROUP_SWITCH_VW1_VW1(blas, F64##KERNEL, SUBGROUP_SUPPORT, DEVICE); \
	} \
	else if(DTYPE == tart::dtypes::complex64) \
	{ \
		SUBGROUP_SWITCH_VW1_VW1(blas, C64##KERNEL, SUBGROUP_SUPPORT, DEVICE); \
	} \
	else if(DTYPE == tart::dtypes::complex128) \
	{ \
		SUBGROUP_SWITCH_VW1_VW1(blas, C128##KERNEL, SUBGROUP_SUPPORT, DEVICE); \
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
		return blas::F16##KERNEL(DEVICE); \
	} \
	else if(DTYPE == tart::dtypes::float32) \
	{ \
		return blas::F32##KERNEL(DEVICE); \
	} \
	else if(DTYPE == tart::dtypes::float64) \
	{ \
		return blas::F64##KERNEL(DEVICE); \
	} \
	else if(DTYPE == tart::dtypes::complex64) \
	{ \
		return blas::C64##KERNEL(DEVICE); \
	} \
	else if(DTYPE == tart::dtypes::complex128) \
	{ \
		return blas::C128##KERNEL(DEVICE); \
	} \
	else \
	{ \
		throw std::runtime_error("dtype not implemented"); \
		return nullptr; \
	}

#endif

tart::kernel_ptr getAmax(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(xamax, dtype, device); }
tart::kernel_ptr getAmaxEpilogue(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(xamax_epilogue, dtype, device); }

tart::kernel_ptr getAsum(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(xasum, dtype, device); }
tart::kernel_ptr getAsumEpilogue(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(xasum_epilogue, dtype, device); }

tart::kernel_ptr getDot(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(xdot, dtype, device); }
tart::kernel_ptr getDotEpilogue(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(xdot_epilogue, dtype, device); }

tart::kernel_ptr getNrm2(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(xnrm2, dtype, device); }
tart::kernel_ptr getNrm2Epilogue(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(xnrm2_epilogue, dtype, device); }

tart::kernel_ptr getCopy(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth) { DTYPE_VW_SWITCH(xcopy, dtype, vectorWidth, device); }
tart::kernel_ptr getCopyFast(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth) { DTYPE_VW_SWITCH(xcopy_fast, dtype, vectorWidth, device); }

tart::kernel_ptr getScal(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth) { DTYPE_VW_SWITCH(xscal, dtype, vectorWidth, device); }
tart::kernel_ptr getScalFast(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth) { DTYPE_VW_SWITCH(xscal_fast, dtype, vectorWidth, device); }

tart::kernel_ptr getSwap(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth) { DTYPE_VW_SWITCH(xswap, dtype, vectorWidth, device); }
tart::kernel_ptr getSwapFast(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth) { DTYPE_VW_SWITCH(xswap_fast, dtype, vectorWidth, device); }

tart::kernel_ptr getAxpy(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth)
{
	DTYPE_VW_SWITCH(xaxpy, dtype, vectorWidth, device);
}

tart::kernel_ptr getAxpyFaster(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth)
{
	DTYPE_VW_SWITCH(xaxpy_faster, dtype, vectorWidth, device);
}

tart::kernel_ptr getAxpyFastest(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth)
{
	DTYPE_VW_SWITCH(xaxpy_fastest, dtype, vectorWidth, device);
}
tart::kernel_ptr getAxpyBatched(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(xaxpy_batched, dtype, device); }

tart::kernel_ptr getHad(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth) { DTYPE_VW_SWITCH(xhad, dtype, vectorWidth, device); }
tart::kernel_ptr getHadFaster(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth) { DTYPE_VW_SWITCH(xhad_faster, dtype, vectorWidth, device); }
tart::kernel_ptr getHadFastest(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth) { DTYPE_VW_SWITCH(xhad_fastest, dtype, vectorWidth, device); }

tart::kernel_ptr getGemv(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(xgemv, dtype, device); }
tart::kernel_ptr getGemvFast(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth) { DTYPE_VW_SWITCH(xgemv_fast, dtype, vectorWidth, device); }
tart::kernel_ptr getGemvFastRot(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth) { DTYPE_VW_SWITCH(xgemv_fast_rot, dtype, vectorWidth, device); }

tart::kernel_ptr getGer(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(xger, dtype, device); }
tart::kernel_ptr getHer(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(xher, dtype, device); }
tart::kernel_ptr getHer2(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(xher2, dtype, device); }
tart::kernel_ptr getTrsvForward(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(xtrsv_forward, dtype, device); }
tart::kernel_ptr getTrsvBackward(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(xtrsv_backward, dtype, device); }
tart::kernel_ptr getTrsvFillVector(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(xtrsv_fill_vector, dtype, device); }

tart::kernel_ptr getGemm(const tart::device_ptr& device, const tart::DType& dtype, const bool subgroupSupport, const uint32_t vwm, const uint32_t vwn)
{
	DTYPE_DUAL_VW_SUBGROUP_SWITCH(dtype, xgemm, subgroupSupport, device)
}
tart::kernel_ptr getGemmUpper(const tart::device_ptr& device, const tart::DType& dtype, const bool subgroupSupport, const uint32_t vwm, const uint32_t vwn)
{
	DTYPE_DUAL_VW_SUBGROUP_SWITCH(dtype, xgemm_upper, subgroupSupport, device)
}
tart::kernel_ptr getGemmLower(const tart::device_ptr& device, const tart::DType& dtype, const bool subgroupSupport, const uint32_t vwm, const uint32_t vwn)
{
	DTYPE_DUAL_VW_SUBGROUP_SWITCH(dtype, xgemm_lower, subgroupSupport, device)
}
tart::kernel_ptr getGemmBatched(const tart::device_ptr& device, const tart::DType& dtype, const bool subgroupSupport, const uint32_t vwm, const uint32_t vwn)
{
	DTYPE_DUAL_VW_SUBGROUP_SWITCH(dtype, xgemm_batched, subgroupSupport, device)
}
tart::kernel_ptr getGemmStridedBatched(const tart::device_ptr& device, const tart::DType& dtype, const bool subgroupSupport, const uint32_t vwm, const uint32_t vwn)
{
	DTYPE_DUAL_VW_SUBGROUP_SWITCH(dtype, xgemm_strided_batched, subgroupSupport, device)
}

tart::kernel_ptr getGemmDirect(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vwm, const uint32_t vwn)
{
	DTYPE_DUAL_VW_SWITCH(dtype, xgemm_direct, vwm, vwn, device)
}
tart::kernel_ptr getGemmDirectBatched(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vwm, const uint32_t vwn)
{
	DTYPE_DUAL_VW_SWITCH(dtype, xgemm_direct_batched, vwm, vwn, device)
}
tart::kernel_ptr getGemmDirectStridedBatched(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vwm, const uint32_t vwn)
{
	DTYPE_DUAL_VW_SWITCH(dtype, xgemm_direct_strided_batched, vwm, vwn, device)
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

tart::kernel_ptr getInvertDiagonalBlock(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(invert_diagonal_block, dtype, device); }
tart::kernel_ptr getInvertTripleMatmul(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(invert_triple_matmul, dtype, device); }
tart::kernel_ptr getInvertTripleMatmulPart2(const tart::device_ptr& device, const tart::DType& dtype) { DTYPE_SWITCH(invert_triple_matmul_part2, dtype, device); }

} // namespace clblast
