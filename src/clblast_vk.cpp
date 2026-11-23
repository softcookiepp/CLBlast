
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements all the BLAS API calls. In all cases, it does not much more than creating
// a new object of the appropriate type, and calling the main routine on that object. It forwards
// all status codes to the caller.
//
// =================================================================================================

#include "clblast_vk.h"

#include <complex>
#include <cstddef>
#include <string>
#include <vector>

#include "database/database.hpp"
#include "routines/routines.hpp"
#include "routines/level2/xgbmv.hpp"
#include "routines/level2/xgemv.hpp"
#include "routines/level2/xger.hpp"
#include "routines/level2/xgerc.hpp"
#include "routines/level2/xgeru.hpp"
#include "routines/level2/xhbmv.hpp"
#include "routines/level2/xhemv.hpp"
#include "routines/level2/xher.hpp"
#include "routines/level2/xher2.hpp"
#include "routines/level2/xhpmv.hpp"
#include "routines/level2/xhpr.hpp"
#include "routines/level2/xhpr2.hpp"
#include "routines/level2/xsbmv.hpp"
#include "routines/level2/xspmv.hpp"
#include "routines/level2/xsyr.hpp"
#include "routines/level2/xspr.hpp"
#include "routines/level2/xsyr2.hpp"
#include "routines/level2/xtbmv.hpp"
#include "routines/level2/xtpmv.hpp"
#include "routines/level2/xtrmv.hpp"
#include "routines/level2/xtrsv.hpp"
#include "routines/level3/xgemm.hpp"
#include "routines/level3/xhemm.hpp"
#include "routines/level3/xher2k.hpp"
#include "routines/level3/xherk.hpp"
#include "routines/level3/xsyr2k.hpp"
#include "routines/level3/xsyrk.hpp"
#include "routines/level3/xsymm.hpp"
#include "routines/level3/xtrmm.hpp"
#include "routines/level3/xtrsm.hpp"
#include "routines/levelx/xaxpybatched.hpp"
#include "routines/levelx/xcol2im.hpp"
#include "routines/levelx/xconvgemm.hpp"
#include "routines/levelx/xgemmbatched.hpp"
#include "routines/levelx/xgemmstridedbatched.hpp"
#include "routines/levelx/xhad.hpp"
#include "routines/levelx/xim2col.hpp"
#include "routines/levelx/xinvert.hpp"
#include "routines/levelx/xomatcopy.hpp"
#include "utilities/backend.hpp"
#include "utilities/clblast_exceptions.hpp"
#include "utilities/utilities.hpp"

namespace clblast {

tart::DType precisionToDataType(Precision p)
{
	switch(p)
	{
		case Precision::kHalf:
			return tart::dtypes::float16;
		case Precision::kSingle:
			return tart::dtypes::float32;
		case Precision::kDouble:
			return tart::dtypes::float64;
		case Precision::kComplexSingle:
			return tart::dtypes::complex64;
		case Precision::kComplexDouble:
			return tart::dtypes::complex128;
		default:
			throw std::runtime_error("invalid precision");
	}
	return tart::dtypes::float32;
}

// =================================================================================================
// BLAS level-1 (vector-vector) routines
// =================================================================================================

// Generate givens plane rotation: SROTG/DROTG
template <typename T>
StatusCode Rotg(tart::buffer_ptr, const size_t, tart::buffer_ptr, const size_t, tart::buffer_ptr, const size_t, tart::buffer_ptr, const size_t) {
	return StatusCode::kNotImplemented;
}
template StatusCode PUBLIC_API Rotg<float>(tart::buffer_ptr, const size_t, tart::buffer_ptr, const size_t, tart::buffer_ptr, const size_t, tart::buffer_ptr,
																					 const size_t);
template StatusCode PUBLIC_API Rotg<double>(tart::buffer_ptr, const size_t, tart::buffer_ptr, const size_t, tart::buffer_ptr, const size_t, tart::buffer_ptr,
																						const size_t);

// Generate modified givens plane rotation: SROTMG/DROTMG
template <typename T>
StatusCode Rotmg(tart::buffer_ptr, const size_t, tart::buffer_ptr, const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t, tart::buffer_ptr,
								 const size_t)
{
	return StatusCode::kNotImplemented;
}
template StatusCode PUBLIC_API Rotmg<float>(tart::buffer_ptr, const size_t, tart::buffer_ptr, const size_t, tart::buffer_ptr, const size_t,
																						const tart::buffer_ptr, const size_t, tart::buffer_ptr, const size_t);
template StatusCode PUBLIC_API Rotmg<double>(tart::buffer_ptr, const size_t, tart::buffer_ptr, const size_t, tart::buffer_ptr, const size_t,
																						 const tart::buffer_ptr, const size_t, tart::buffer_ptr, const size_t);

// Apply givens plane rotation: SROT/DROT
template <typename T>
StatusCode Rot(const size_t, tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr, const size_t, const size_t, const T, const T) {
	return StatusCode::kNotImplemented;
}
template StatusCode PUBLIC_API Rot<float>(const size_t, tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr, const size_t,
																					const size_t, const float, const float);
template StatusCode PUBLIC_API Rot<double>(const size_t, tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr, const size_t,
																					 const size_t, const double, const double);

// Apply modified givens plane rotation: SROTM/DROTM
template <typename T>
StatusCode Rotm(const size_t, tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr,
								const size_t) {
	return StatusCode::kNotImplemented;
}
template StatusCode PUBLIC_API Rotm<float>(const size_t, tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr, const size_t,
																					 const size_t, tart::buffer_ptr, const size_t);
template StatusCode PUBLIC_API Rotm<double>(const size_t, tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr, const size_t,
																						const size_t, tart::buffer_ptr, const size_t);

// Swap two vectors: SSWAP/DSWAP/CSWAP/ZSWAP/HSWAP
template <typename T>
StatusCode Swap(const size_t n, tart::buffer_ptr x_buffer, const size_t x_offset, const size_t x_inc, tart::buffer_ptr y_buffer,
								const size_t y_offset, const size_t y_inc)
{
	try
	{
		auto device = x_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Xaxpy"}, dtype);
		DoSwap(dtype, device, n, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, db);
		return StatusCode::kSuccess;
	}
	catch (...)
	{
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Swap<float>(const size_t, tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr, const size_t,
																					 const size_t);
template StatusCode PUBLIC_API Swap<double>(const size_t, tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr, const size_t,
																						const size_t);
template StatusCode PUBLIC_API Swap<float2>(const size_t, tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr, const size_t,
																						const size_t);
template StatusCode PUBLIC_API Swap<double2>(const size_t, tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr, const size_t,
																						 const size_t);
template StatusCode PUBLIC_API Swap<half>(const size_t, tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr, const size_t,
																					const size_t);

// Vector scaling: SSCAL/DSCAL/CSCAL/ZSCAL/HSCAL
template <typename T>
StatusCode Scal(const size_t n, const T alpha, tart::buffer_ptr x_buffer, const size_t x_offset, const size_t x_inc) {
	try {
		auto device = x_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Xaxpy"}, dtype);
		DoScal(dtype, device, n, alpha, x_buffer, x_offset, x_inc, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Scal<float>(const size_t, const float, tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Scal<double>(const size_t, const double, tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Scal<float2>(const size_t, const float2, tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Scal<double2>(const size_t, const double2, tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Scal<half>(const size_t, const half, tart::buffer_ptr, const size_t, const size_t);

// Vector copy: SCOPY/DCOPY/CCOPY/ZCOPY/HCOPY
template <typename T>
StatusCode Copy(const size_t n, const tart::buffer_ptr x_buffer, const size_t x_offset, const size_t x_inc, tart::buffer_ptr y_buffer,
								const size_t y_offset, const size_t y_inc) {
	try {
		auto device = x_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Xaxpy"}, dtype);
		DoCopy(dtype, device, n, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Copy<float>(const size_t, const tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr, const size_t,
																					 const size_t);
template StatusCode PUBLIC_API Copy<double>(const size_t, const tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr,
																						const size_t, const size_t);
template StatusCode PUBLIC_API Copy<float2>(const size_t, const tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr,
																						const size_t, const size_t);
template StatusCode PUBLIC_API Copy<double2>(const size_t, const tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr,
																						 const size_t, const size_t);
template StatusCode PUBLIC_API Copy<half>(const size_t, const tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr, const size_t,
																					const size_t);

// Vector-times-constant plus vector: SAXPY/DAXPY/CAXPY/ZAXPY/HAXPY
template <typename T>
StatusCode Axpy(const size_t n, const T alpha, const tart::buffer_ptr x_buffer, const size_t x_offset, const size_t x_inc,
								tart::buffer_ptr y_buffer, const size_t y_offset, const size_t y_inc) {
	try {
		auto device = x_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Xaxpy"}, dtype);
		DoAxpy(dtype, device, n, alpha, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Axpy<float>(const size_t, const float, const tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr,
																					 const size_t, const size_t);
template StatusCode PUBLIC_API Axpy<double>(const size_t, const double, const tart::buffer_ptr, const size_t, const size_t,
																						tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Axpy<float2>(const size_t, const float2, const tart::buffer_ptr, const size_t, const size_t,
																						tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Axpy<double2>(const size_t, const double2, const tart::buffer_ptr, const size_t, const size_t,
																						 tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Axpy<half>(const size_t, const half, const tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr,
																					const size_t, const size_t);

// Dot product of two vectors: SDOT/DDOT/HDOT
template <typename T>
StatusCode Dot(const size_t n, tart::buffer_ptr dot_buffer, const size_t dot_offset, const tart::buffer_ptr x_buffer, const size_t x_offset,
							 const size_t x_inc, const tart::buffer_ptr y_buffer, const size_t y_offset, const size_t y_inc) {
	try {
		auto device = dot_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Xdot"}, dtype);
		DoDot(dtype, device, n, dot_buffer, dot_offset, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, false, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Dot<float>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t, const size_t,
																					const tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Dot<double>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t, const size_t,
																					 const tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Dot<half>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t, const size_t,
																				 const tart::buffer_ptr, const size_t, const size_t);

// Dot product of two complex vectors: CDOTU/ZDOTU
template <typename T>
StatusCode Dotu(const size_t n, tart::buffer_ptr dot_buffer, const size_t dot_offset, const tart::buffer_ptr x_buffer,
								const size_t x_offset, const size_t x_inc, const tart::buffer_ptr y_buffer, const size_t y_offset,
								const size_t y_inc) {
	try
	{
		auto device = dot_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Xdot"}, dtype);
		DoDotu(dtype, device, n, dot_buffer, dot_offset, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, db);
		return StatusCode::kSuccess;
	}
	catch (...)
	{
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Dotu<float2>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t,
																						const size_t, const tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Dotu<double2>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t,
																						 const size_t, const tart::buffer_ptr, const size_t, const size_t);

// Dot product of two complex vectors, one conjugated: CDOTC/ZDOTC
template <typename T>
StatusCode Dotc(const size_t n, tart::buffer_ptr dot_buffer, const size_t dot_offset, const tart::buffer_ptr x_buffer,
								const size_t x_offset, const size_t x_inc, const tart::buffer_ptr y_buffer, const size_t y_offset,
								const size_t y_inc) {
	try {
		auto device = dot_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Xdot"}, dtype);
		DoDotc(dtype, device, n, dot_buffer, dot_offset, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Dotc<float2>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t,
																						const size_t, const tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Dotc<double2>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t,
																						 const size_t, const tart::buffer_ptr, const size_t, const size_t);

// Euclidian norm of a vector: SNRM2/DNRM2/ScNRM2/DzNRM2/HNRM2
template <typename T>
StatusCode Nrm2(const size_t n, tart::buffer_ptr nrm2_buffer, const size_t nrm2_offset, const tart::buffer_ptr x_buffer,
								const size_t x_offset, const size_t x_inc) {
	try {
		auto device = nrm2_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Xdot"}, dtype);
		DoNrm2(dtype, device, n, nrm2_buffer, nrm2_offset, x_buffer, x_offset, x_inc, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Nrm2<float>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Nrm2<double>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t,
																						const size_t);
template StatusCode PUBLIC_API Nrm2<float2>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t,
																						const size_t);
template StatusCode PUBLIC_API Nrm2<double2>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t,
																						 const size_t);
template StatusCode PUBLIC_API Nrm2<half>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t, const size_t);

// Absolute sum of values in a vector: SASUM/DASUM/ScASUM/DzASUM/HASUM
template <typename T>
StatusCode Asum(const size_t n, tart::buffer_ptr asum_buffer, const size_t asum_offset, const tart::buffer_ptr x_buffer,
								const size_t x_offset, const size_t x_inc) {
	try {
		auto device = asum_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Xdot"}, dtype);
		DoAsum(dtype, device, n, asum_buffer, asum_offset, x_buffer, x_offset, x_inc, 0, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Asum<float>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Asum<double>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t,
																						const size_t);
template StatusCode PUBLIC_API Asum<float2>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t,
																						const size_t);
template StatusCode PUBLIC_API Asum<double2>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t,
																						 const size_t);
template StatusCode PUBLIC_API Asum<half>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t, const size_t);

// Sum of values in a vector (non-BLAS function): SSUM/DSUM/ScSUM/DzSUM/HSUM
template <typename T>
StatusCode Sum(const size_t n, tart::buffer_ptr sum_buffer, const size_t sum_offset, const tart::buffer_ptr x_buffer, const size_t x_offset,
							 const size_t x_inc) {
	try {
		auto device = sum_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Xdot"}, dtype);
		DoSum(dtype, device, n, sum_buffer, sum_offset, x_buffer, x_offset, x_inc, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Sum<float>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Sum<double>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Sum<float2>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Sum<double2>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t,
																						const size_t);
template StatusCode PUBLIC_API Sum<half>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t, const size_t);

// Index of absolute maximum value in a vector: iSAMAX/iDAMAX/iCAMAX/iZAMAX/iHAMAX
template <typename T>
StatusCode Amax(const size_t n, tart::buffer_ptr imax_buffer, const size_t imax_offset, const tart::buffer_ptr x_buffer,
								const size_t x_offset, const size_t x_inc) {
	try {
		auto device = imax_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Xdot"}, dtype);
		DoAmax(dtype, device, n, imax_buffer, imax_offset, x_buffer, x_offset, x_inc, 1, 0, 0, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Amax<float>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Amax<double>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t,
																						const size_t);
template StatusCode PUBLIC_API Amax<float2>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t,
																						const size_t);
template StatusCode PUBLIC_API Amax<double2>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t,
																						 const size_t);
template StatusCode PUBLIC_API Amax<half>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t, const size_t);

// Index of absolute minimum value in a vector (non-BLAS function): iSAMIN/iDAMIN/iCAMIN/iZAMIN/iHAMIN
template <typename T>
StatusCode Amin(const size_t n, tart::buffer_ptr imin_buffer, const size_t imin_offset, const tart::buffer_ptr x_buffer,
								const size_t x_offset, const size_t x_inc) {
	try {
		auto device = imin_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Xdot"}, dtype);
		DoAmin(dtype, device, n, imin_buffer, imin_offset, x_buffer, x_offset, x_inc, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Amin<float>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Amin<double>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t,
																						const size_t);
template StatusCode PUBLIC_API Amin<float2>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t,
																						const size_t);
template StatusCode PUBLIC_API Amin<double2>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t,
																						 const size_t);
template StatusCode PUBLIC_API Amin<half>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t, const size_t);

// Index of maximum value in a vector (non-BLAS function): iSMAX/iDMAX/iCMAX/iZMAX/iHMAX
template <typename T>
StatusCode Max(const size_t n, tart::buffer_ptr imax_buffer, const size_t imax_offset, const tart::buffer_ptr x_buffer,
							 const size_t x_offset, const size_t x_inc) {
	try {
		auto device = imax_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Xdot"}, dtype);
		DoMax(dtype, device, n, imax_buffer, imax_offset, x_buffer, x_offset, x_inc, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Max<float>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Max<double>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Max<float2>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Max<double2>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t,
																						const size_t);
template StatusCode PUBLIC_API Max<half>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t, const size_t);

// Index of minimum value in a vector (non-BLAS function): iSMIN/iDMIN/iCMIN/iZMIN/iHMIN
template <typename T>
StatusCode Min(const size_t n, tart::buffer_ptr imin_buffer, const size_t imin_offset, const tart::buffer_ptr x_buffer,
							 const size_t x_offset, const size_t x_inc) {
	try {
		auto device = imin_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Xdot"}, dtype);
		DoMin(dtype, device, n, imin_buffer, imin_offset, x_buffer, x_offset, x_inc, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Min<float>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Min<double>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Min<float2>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Min<double2>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t,
																						const size_t);
template StatusCode PUBLIC_API Min<half>(const size_t, tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t, const size_t);

// =================================================================================================
// BLAS level-2 (matrix-vector) routines
// =================================================================================================

// General matrix-vector multiplication: SGEMV/DGEMV/CGEMV/ZGEMV/HGEMV
template <typename T>
StatusCode Gemv(const Layout layout, const Transpose a_transpose, const size_t m, const size_t n, const T alpha,
								const tart::buffer_ptr a_buffer, const size_t a_offset, const size_t a_ld, const tart::buffer_ptr x_buffer,
								const size_t x_offset, const size_t x_inc, const T beta, tart::buffer_ptr y_buffer, const size_t y_offset,
								const size_t y_inc) {
	try {
		auto device = a_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Gemm"}, dtype);
		DoGemv(layout, a_transpose, m, n, alpha, a_buffer, a_offset, a_ld, x_buffer, x_offset,
					 x_inc, beta, y_buffer, y_offset, y_inc, db, device, dtype);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Gemv<float>(const Layout, const Transpose, const size_t, const size_t, const float,
																					 const tart::buffer_ptr, const size_t, const size_t, const tart::buffer_ptr, const size_t,
																					 const size_t, const float, tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Gemv<double>(const Layout, const Transpose, const size_t, const size_t, const double,
																						const tart::buffer_ptr, const size_t, const size_t, const tart::buffer_ptr, const size_t,
																						const size_t, const double, tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Gemv<float2>(const Layout, const Transpose, const size_t, const size_t, const float2,
																						const tart::buffer_ptr, const size_t, const size_t, const tart::buffer_ptr, const size_t,
																						const size_t, const float2, tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Gemv<double2>(const Layout, const Transpose, const size_t, const size_t, const double2,
																						 const tart::buffer_ptr, const size_t, const size_t, const tart::buffer_ptr, const size_t,
																						 const size_t, const double2, tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Gemv<half>(const Layout, const Transpose, const size_t, const size_t, const half,
																					const tart::buffer_ptr, const size_t, const size_t, const tart::buffer_ptr, const size_t,
																					const size_t, const half, tart::buffer_ptr, const size_t, const size_t);

// General banded matrix-vector multiplication: SGBMV/DGBMV/CGBMV/ZGBMV/HGBMV
template <typename T>
StatusCode Gbmv(const Layout layout, const Transpose a_transpose, const size_t m, const size_t n, const size_t kl,
								const size_t ku, const T alpha, const tart::buffer_ptr a_buffer, const size_t a_offset, const size_t a_ld,
								const tart::buffer_ptr x_buffer, const size_t x_offset, const size_t x_inc, const T beta, tart::buffer_ptr y_buffer,
								const size_t y_offset, const size_t y_inc) {
	try {
		auto device = a_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Gemm"}, dtype);
		auto routine = Xgbmv(dtype, queue_cpp);
		routine.DoGbmv(layout, a_transpose, m, n, kl, ku, alpha, a_buffer, a_offset, a_ld, x_buffer,
									 x_offset, x_inc, beta, y_buffer, y_offset, y_inc, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Gbmv<float>(const Layout, const Transpose, const size_t, const size_t, const size_t,
																					 const size_t, const float, const tart::buffer_ptr, const size_t, const size_t,
																					 const tart::buffer_ptr, const size_t, const size_t, const float, tart::buffer_ptr, const size_t,
																					 const size_t);
template StatusCode PUBLIC_API Gbmv<double>(const Layout, const Transpose, const size_t, const size_t, const size_t,
																						const size_t, const double, const tart::buffer_ptr, const size_t, const size_t,
																						const tart::buffer_ptr, const size_t, const size_t, const double, tart::buffer_ptr,
																						const size_t, const size_t);
template StatusCode PUBLIC_API Gbmv<float2>(const Layout, const Transpose, const size_t, const size_t, const size_t,
																						const size_t, const float2, const tart::buffer_ptr, const size_t, const size_t,
																						const tart::buffer_ptr, const size_t, const size_t, const float2, tart::buffer_ptr,
																						const size_t, const size_t);
template StatusCode PUBLIC_API Gbmv<double2>(const Layout, const Transpose, const size_t, const size_t, const size_t,
																						 const size_t, const double2, const tart::buffer_ptr, const size_t, const size_t,
																						 const tart::buffer_ptr, const size_t, const size_t, const double2, tart::buffer_ptr,
																						 const size_t, const size_t);
template StatusCode PUBLIC_API Gbmv<half>(const Layout, const Transpose, const size_t, const size_t, const size_t,
																					const size_t, const half, const tart::buffer_ptr, const size_t, const size_t,
																					const tart::buffer_ptr, const size_t, const size_t, const half, tart::buffer_ptr, const size_t,
																					const size_t);

// Hermitian matrix-vector multiplication: CHEMV/ZHEMV
template <typename T>
StatusCode Hemv(const Layout layout, const Triangle triangle, const size_t n, const T alpha, const tart::buffer_ptr a_buffer,
								const size_t a_offset, const size_t a_ld, const tart::buffer_ptr x_buffer, const size_t x_offset,
								const size_t x_inc, const T beta, tart::buffer_ptr y_buffer, const size_t y_offset, const size_t y_inc) {
	try {
		auto device = a_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Gemm"}, dtype);
		auto routine = Xhemv(dtype, queue_cpp);
		routine.DoHemv(layout, triangle, n, alpha, a_buffer, a_offset, a_ld, x_buffer, x_offset,
									 x_inc, beta, y_buffer, y_offset, y_inc, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Hemv<float2>(const Layout, const Triangle, const size_t, const float2, const tart::buffer_ptr,
																						const size_t, const size_t, const tart::buffer_ptr, const size_t, const size_t,
																						const float2, tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Hemv<double2>(const Layout, const Triangle, const size_t, const double2, const tart::buffer_ptr,
																						 const size_t, const size_t, const tart::buffer_ptr, const size_t, const size_t,
																						 const double2, tart::buffer_ptr, const size_t, const size_t);

// Hermitian banded matrix-vector multiplication: CHBMV/ZHBMV
template <typename T>
StatusCode Hbmv(const Layout layout, const Triangle triangle, const size_t n, const size_t k, const T alpha,
								const tart::buffer_ptr a_buffer, const size_t a_offset, const size_t a_ld, const tart::buffer_ptr x_buffer,
								const size_t x_offset, const size_t x_inc, const T beta, tart::buffer_ptr y_buffer, const size_t y_offset,
								const size_t y_inc) {
	try {
		auto device = a_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Gemm"}, dtype);
		auto routine = Xhbmv(dtype, queue_cpp);
		routine.DoHbmv(layout, triangle, n, k, alpha, a_buffer, a_offset, a_ld, x_buffer, x_offset,
									 x_inc, beta, y_buffer, y_offset, y_inc, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Hbmv<float2>(const Layout, const Triangle, const size_t, const size_t, const float2,
																						const tart::buffer_ptr, const size_t, const size_t, const tart::buffer_ptr, const size_t,
																						const size_t, const float2, tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Hbmv<double2>(const Layout, const Triangle, const size_t, const size_t, const double2,
																						 const tart::buffer_ptr, const size_t, const size_t, const tart::buffer_ptr, const size_t,
																						 const size_t, const double2, tart::buffer_ptr, const size_t, const size_t);

// Hermitian packed matrix-vector multiplication: CHPMV/ZHPMV
template <typename T>
StatusCode Hpmv(const Layout layout, const Triangle triangle, const size_t n, const T alpha, const tart::buffer_ptr ap_buffer,
								const size_t ap_offset, const tart::buffer_ptr x_buffer, const size_t x_offset, const size_t x_inc, const T beta,
								tart::buffer_ptr y_buffer, const size_t y_offset, const size_t y_inc) {
	try {
		auto device = ap_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Gemm"}, dtype);
		auto routine = Xhpmv(dtype, queue_cpp);
		routine.DoHpmv(layout, triangle, n, alpha, ap_buffer, ap_offset, x_buffer, x_offset, x_inc,
									 beta, y_buffer, y_offset, y_inc, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Hpmv<float2>(const Layout, const Triangle, const size_t, const float2, const tart::buffer_ptr,
																						const size_t, const tart::buffer_ptr, const size_t, const size_t, const float2,
																						tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Hpmv<double2>(const Layout, const Triangle, const size_t, const double2, const tart::buffer_ptr,
																						 const size_t, const tart::buffer_ptr, const size_t, const size_t, const double2,
																						 tart::buffer_ptr, const size_t, const size_t);

// Symmetric matrix-vector multiplication: SSYMV/DSYMV/HSYMV
template <typename T>
StatusCode Symv(const Layout layout, const Triangle triangle, const size_t n, const T alpha, const tart::buffer_ptr a_buffer,
								const size_t a_offset, const size_t a_ld, const tart::buffer_ptr x_buffer, const size_t x_offset,
								const size_t x_inc, const T beta, tart::buffer_ptr y_buffer, const size_t y_offset, const size_t y_inc) {
	try {
		auto device = a_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Gemm"}, dtype);
		auto routine = Xsymv(dtype, queue_cpp);
		routine.DoSymv(layout, triangle, n, alpha, a_buffer, a_offset, a_ld, x_buffer, x_offset,
									 x_inc, beta, y_buffer, y_offset, y_inc, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Symv<float>(const Layout, const Triangle, const size_t, const float, const tart::buffer_ptr,
																					 const size_t, const size_t, const tart::buffer_ptr, const size_t, const size_t,
																					 const float, tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Symv<double>(const Layout, const Triangle, const size_t, const double, const tart::buffer_ptr,
																						const size_t, const size_t, const tart::buffer_ptr, const size_t, const size_t,
																						const double, tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Symv<half>(const Layout, const Triangle, const size_t, const half, const tart::buffer_ptr,
																					const size_t, const size_t, const tart::buffer_ptr, const size_t, const size_t,
																					const half, tart::buffer_ptr, const size_t, const size_t);

// Symmetric banded matrix-vector multiplication: SSBMV/DSBMV/HSBMV
template <typename T>
StatusCode Sbmv(const Layout layout, const Triangle triangle, const size_t n, const size_t k, const T alpha,
								const tart::buffer_ptr a_buffer, const size_t a_offset, const size_t a_ld, const tart::buffer_ptr x_buffer,
								const size_t x_offset, const size_t x_inc, const T beta, tart::buffer_ptr y_buffer, const size_t y_offset,
								const size_t y_inc) {
	try {
		auto device = a_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Gemm"}, dtype);
		auto routine = Xsbmv(dtype, queue_cpp);
		routine.DoSbmv(layout, triangle, n, k, alpha, a_buffer, a_offset, a_ld, x_buffer, x_offset,
									 x_inc, beta, y_buffer, y_offset, y_inc, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Sbmv<float>(const Layout, const Triangle, const size_t, const size_t, const float,
																					 const tart::buffer_ptr, const size_t, const size_t, const tart::buffer_ptr, const size_t,
																					 const size_t, const float, tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Sbmv<double>(const Layout, const Triangle, const size_t, const size_t, const double,
																						const tart::buffer_ptr, const size_t, const size_t, const tart::buffer_ptr, const size_t,
																						const size_t, const double, tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Sbmv<half>(const Layout, const Triangle, const size_t, const size_t, const half,
																					const tart::buffer_ptr, const size_t, const size_t, const tart::buffer_ptr, const size_t,
																					const size_t, const half, tart::buffer_ptr, const size_t, const size_t);

// Symmetric packed matrix-vector multiplication: SSPMV/DSPMV/HSPMV
template <typename T>
StatusCode Spmv(const Layout layout, const Triangle triangle, const size_t n, const T alpha, const tart::buffer_ptr ap_buffer,
								const size_t ap_offset, const tart::buffer_ptr x_buffer, const size_t x_offset, const size_t x_inc, const T beta,
								tart::buffer_ptr y_buffer, const size_t y_offset, const size_t y_inc) {
	try {
		auto device = ap_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Gemm"}, dtype);
		auto routine = Xspmv(dtype, queue_cpp);
		routine.DoSpmv(layout, triangle, n, alpha, ap_buffer, ap_offset, x_buffer, x_offset, x_inc,
									 beta, y_buffer, y_offset, y_inc, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Spmv<float>(const Layout, const Triangle, const size_t, const float, const tart::buffer_ptr,
																					 const size_t, const tart::buffer_ptr, const size_t, const size_t, const float, tart::buffer_ptr,
																					 const size_t, const size_t);
template StatusCode PUBLIC_API Spmv<double>(const Layout, const Triangle, const size_t, const double, const tart::buffer_ptr,
																						const size_t, const tart::buffer_ptr, const size_t, const size_t, const double,
																						tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Spmv<half>(const Layout, const Triangle, const size_t, const half, const tart::buffer_ptr,
																					const size_t, const tart::buffer_ptr, const size_t, const size_t, const half, tart::buffer_ptr,
																					const size_t, const size_t);

// Triangular matrix-vector multiplication: STRMV/DTRMV/CTRMV/ZTRMV/HTRMV
template <typename T>
StatusCode Trmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
								const size_t n, const tart::buffer_ptr a_buffer, const size_t a_offset, const size_t a_ld, tart::buffer_ptr x_buffer,
								const size_t x_offset, const size_t x_inc) {
	try {
		auto device = a_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Gemm"}, dtype);
		auto routine = Xtrmv(dtype, queue_cpp);
		routine.DoTrmv(layout, triangle, a_transpose, diagonal, n, a_buffer, a_offset, a_ld, x_buffer,
									 x_offset, x_inc, device, dtype, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Trmv<float>(const Layout, const Triangle, const Transpose, const Diagonal, const size_t,
																					 const tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Trmv<double>(const Layout, const Triangle, const Transpose, const Diagonal, const size_t,
																						const tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr, const size_t,
																						const size_t);
template StatusCode PUBLIC_API Trmv<float2>(const Layout, const Triangle, const Transpose, const Diagonal, const size_t,
																						const tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr, const size_t,
																						const size_t);
template StatusCode PUBLIC_API Trmv<double2>(const Layout, const Triangle, const Transpose, const Diagonal,
																						 const size_t, const tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr,
																						 const size_t, const size_t);
template StatusCode PUBLIC_API Trmv<half>(const Layout, const Triangle, const Transpose, const Diagonal, const size_t,
																					const tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr, const size_t, const size_t);

// Triangular banded matrix-vector multiplication: STBMV/DTBMV/CTBMV/ZTBMV/HTBMV
template <typename T>
StatusCode Tbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
								const size_t n, const size_t k, const tart::buffer_ptr a_buffer, const size_t a_offset, const size_t a_ld,
								tart::buffer_ptr x_buffer, const size_t x_offset, const size_t x_inc) {
	try {
		auto device = a_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Gemm"}, dtype);
		auto routine = Xtbmv(dtype, queue_cpp);
		routine.DoTbmv(layout, triangle, a_transpose, diagonal, n, k, a_buffer, a_offset, a_ld,
			x_buffer, x_offset, x_inc, device, dtype, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Tbmv<float>(const Layout, const Triangle, const Transpose, const Diagonal, const size_t,
																					 const size_t, const tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr, const size_t,
																					 const size_t);
template StatusCode PUBLIC_API Tbmv<double>(const Layout, const Triangle, const Transpose, const Diagonal, const size_t,
																						const size_t, const tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr,
																						const size_t, const size_t);
template StatusCode PUBLIC_API Tbmv<float2>(const Layout, const Triangle, const Transpose, const Diagonal, const size_t,
																						const size_t, const tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr,
																						const size_t, const size_t);
template StatusCode PUBLIC_API Tbmv<double2>(const Layout, const Triangle, const Transpose, const Diagonal,
																						 const size_t, const size_t, const tart::buffer_ptr, const size_t, const size_t,
																						 tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Tbmv<half>(const Layout, const Triangle, const Transpose, const Diagonal, const size_t,
																					const size_t, const tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr, const size_t,
																					const size_t);

// Triangular packed matrix-vector multiplication: STPMV/DTPMV/CTPMV/ZTPMV/HTPMV
template <typename T>
StatusCode Tpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
								const size_t n, const tart::buffer_ptr ap_buffer, const size_t ap_offset, tart::buffer_ptr x_buffer, const size_t x_offset,
								const size_t x_inc) {
	try {
		auto device = ap_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Gemm"}, dtype);
		auto routine = Xtpmv(dtype, queue_cpp);
		routine.DoTpmv(layout, triangle, a_transpose, diagonal, n, ap_buffer, ap_offset, x_buffer,
									 x_offset, x_inc, device, dtype, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Tpmv<float>(const Layout, const Triangle, const Transpose, const Diagonal, const size_t,
																					 const tart::buffer_ptr, const size_t, tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Tpmv<double>(const Layout, const Triangle, const Transpose, const Diagonal, const size_t,
																						const tart::buffer_ptr, const size_t, tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Tpmv<float2>(const Layout, const Triangle, const Transpose, const Diagonal, const size_t,
																						const tart::buffer_ptr, const size_t, tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Tpmv<double2>(const Layout, const Triangle, const Transpose, const Diagonal,
																						 const size_t, const tart::buffer_ptr, const size_t, tart::buffer_ptr, const size_t,
																						 const size_t);
template StatusCode PUBLIC_API Tpmv<half>(const Layout, const Triangle, const Transpose, const Diagonal, const size_t,
																					const tart::buffer_ptr, const size_t, tart::buffer_ptr, const size_t, const size_t);

// Solves a triangular system of equations: STRSV/DTRSV/CTRSV/ZTRSV
template <typename T>
StatusCode Trsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
								const size_t n, const tart::buffer_ptr a_buffer, const size_t a_offset, const size_t a_ld, tart::buffer_ptr x_buffer,
								const size_t x_offset, const size_t x_inc) {
	try {
		auto device = a_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Trsv"}, dtype);
		auto routine = Xtrsv(dtype, queue_cpp);
		routine.DoTrsv(layout, triangle, a_transpose, diagonal, n, a_buffer, a_offset, a_ld, x_buffer,
									 x_offset, x_inc, device, dtype, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Trsv<float>(const Layout, const Triangle, const Transpose, const Diagonal, const size_t,
																					 const tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Trsv<double>(const Layout, const Triangle, const Transpose, const Diagonal, const size_t,
																						const tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr, const size_t,
																						const size_t);
template StatusCode PUBLIC_API Trsv<float2>(const Layout, const Triangle, const Transpose, const Diagonal, const size_t,
																						const tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr, const size_t,
																						const size_t);
template StatusCode PUBLIC_API Trsv<double2>(const Layout, const Triangle, const Transpose, const Diagonal,
																						 const size_t, const tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr,
																						 const size_t, const size_t);

// Solves a banded triangular system of equations: STBSV/DTBSV/CTBSV/ZTBSV
template <typename T>
StatusCode Tbsv(const Layout, const Triangle, const Transpose, const Diagonal, const size_t, const size_t, const tart::buffer_ptr,
								const size_t, const size_t, tart::buffer_ptr, const size_t, const size_t) {
	return StatusCode::kNotImplemented;
}
template StatusCode PUBLIC_API Tbsv<float>(const Layout, const Triangle, const Transpose, const Diagonal, const size_t,
																					 const size_t, const tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr, const size_t,
																					 const size_t);
template StatusCode PUBLIC_API Tbsv<double>(const Layout, const Triangle, const Transpose, const Diagonal, const size_t,
																						const size_t, const tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr,
																						const size_t, const size_t);
template StatusCode PUBLIC_API Tbsv<float2>(const Layout, const Triangle, const Transpose, const Diagonal, const size_t,
																						const size_t, const tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr,
																						const size_t, const size_t);
template StatusCode PUBLIC_API Tbsv<double2>(const Layout, const Triangle, const Transpose, const Diagonal,
																						 const size_t, const size_t, const tart::buffer_ptr, const size_t, const size_t,
																						 tart::buffer_ptr, const size_t, const size_t);

// Solves a packed triangular system of equations: STPSV/DTPSV/CTPSV/ZTPSV
template <typename T>
StatusCode Tpsv(const Layout, const Triangle, const Transpose, const Diagonal, const size_t, const tart::buffer_ptr, const size_t,
								tart::buffer_ptr, const size_t, const size_t) {
	return StatusCode::kNotImplemented;
}
template StatusCode PUBLIC_API Tpsv<float>(const Layout, const Triangle, const Transpose, const Diagonal, const size_t,
																					 const tart::buffer_ptr, const size_t, tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Tpsv<double>(const Layout, const Triangle, const Transpose, const Diagonal, const size_t,
																						const tart::buffer_ptr, const size_t, tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Tpsv<float2>(const Layout, const Triangle, const Transpose, const Diagonal, const size_t,
																						const tart::buffer_ptr, const size_t, tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Tpsv<double2>(const Layout, const Triangle, const Transpose, const Diagonal,
																						 const size_t, const tart::buffer_ptr, const size_t, tart::buffer_ptr, const size_t,
																						 const size_t);

// General rank-1 matrix update: SGER/DGER/HGER
template <typename T>
StatusCode Ger(const Layout layout, const size_t m, const size_t n, const T alpha, const tart::buffer_ptr x_buffer,
							 const size_t x_offset, const size_t x_inc, const tart::buffer_ptr y_buffer, const size_t y_offset,
							 const size_t y_inc, tart::buffer_ptr a_buffer, const size_t a_offset, const size_t a_ld) {
	try {
		auto device = x_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Ger"}, dtype);
		auto routine = Xger(dtype, queue_cpp);
		routine.DoGer(layout, m, n, alpha, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc,
									a_buffer, a_offset, a_ld, "GER", db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Ger<float>(const Layout, const size_t, const size_t, const float, const tart::buffer_ptr,
																					const size_t, const size_t, const tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr,
																					const size_t, const size_t);
template StatusCode PUBLIC_API Ger<double>(const Layout, const size_t, const size_t, const double, const tart::buffer_ptr,
																					 const size_t, const size_t, const tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr,
																					 const size_t, const size_t);
template StatusCode PUBLIC_API Ger<half>(const Layout, const size_t, const size_t, const half, const tart::buffer_ptr,
																				 const size_t, const size_t, const tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr,
																				 const size_t, const size_t);

// General rank-1 complex matrix update: CGERU/ZGERU
template <typename T>
StatusCode Geru(const Layout layout, const size_t m, const size_t n, const T alpha, const tart::buffer_ptr x_buffer,
								const size_t x_offset, const size_t x_inc, const tart::buffer_ptr y_buffer, const size_t y_offset,
								const size_t y_inc, tart::buffer_ptr a_buffer, const size_t a_offset, const size_t a_ld) {
	try {
		auto device = x_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Ger"}, dtype);
		auto routine = Xgeru(dtype, queue_cpp);
		routine.DoGeru(layout, m, n, alpha, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc,
									 a_buffer, a_offset, a_ld, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Geru<float2>(const Layout, const size_t, const size_t, const float2, const tart::buffer_ptr,
																						const size_t, const size_t, const tart::buffer_ptr, const size_t, const size_t,
																						tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Geru<double2>(const Layout, const size_t, const size_t, const double2, const tart::buffer_ptr,
																						 const size_t, const size_t, const tart::buffer_ptr, const size_t, const size_t,
																						 tart::buffer_ptr, const size_t, const size_t);

// General rank-1 complex conjugated matrix update: CGERC/ZGERC
template <typename T>
StatusCode Gerc(const Layout layout, const size_t m, const size_t n, const T alpha, const tart::buffer_ptr x_buffer,
								const size_t x_offset, const size_t x_inc, const tart::buffer_ptr y_buffer, const size_t y_offset,
								const size_t y_inc, tart::buffer_ptr a_buffer, const size_t a_offset, const size_t a_ld) {
	try {
		auto device = x_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Ger"}, dtype);
		auto routine = Xgerc(dtype, queue_cpp);
		routine.DoGerc(layout, m, n, alpha, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc,
									 a_buffer, a_offset, a_ld, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Gerc<float2>(const Layout, const size_t, const size_t, const float2, const tart::buffer_ptr,
																						const size_t, const size_t, const tart::buffer_ptr, const size_t, const size_t,
																						tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Gerc<double2>(const Layout, const size_t, const size_t, const double2, const tart::buffer_ptr,
																						 const size_t, const size_t, const tart::buffer_ptr, const size_t, const size_t,
																						 tart::buffer_ptr, const size_t, const size_t);

// Hermitian rank-1 matrix update: CHER/ZHER
template <typename T>
StatusCode Her(const Layout layout, const Triangle triangle, const size_t n, const T alpha, const tart::buffer_ptr x_buffer,
							 const size_t x_offset, const size_t x_inc, tart::buffer_ptr a_buffer, const size_t a_offset, const size_t a_ld) {
	try {
		auto device = x_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<std::complex<T>>();
		auto db = GetDatabases(device, {"Her"}, dtype);
		auto routine = Xher(dtype, queue_cpp);
		routine.DoHer(layout, triangle, n, alpha, x_buffer, x_offset, x_inc,
									a_buffer, a_offset, a_ld, false, "HER", db, device, dtype);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Her<float>(const Layout, const Triangle, const size_t, const float, const tart::buffer_ptr,
																					const size_t, const size_t, tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Her<double>(const Layout, const Triangle, const size_t, const double, const tart::buffer_ptr,
																					 const size_t, const size_t, tart::buffer_ptr, const size_t, const size_t);

// Hermitian packed rank-1 matrix update: CHPR/ZHPR
template <typename T>
StatusCode Hpr(const Layout layout, const Triangle triangle, const size_t n, const T alpha, const tart::buffer_ptr x_buffer,
							 const size_t x_offset, const size_t x_inc, tart::buffer_ptr ap_buffer, const size_t ap_offset) {
	try {
		auto device = x_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<std::complex<T>>();
		auto db = GetDatabases(device, {"Her"}, dtype);
		auto routine = Xhpr(dtype, queue_cpp);
		routine.DoHpr(layout, triangle, n, alpha, x_buffer, x_offset, x_inc,
									ap_buffer, ap_offset, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Hpr<float>(const Layout, const Triangle, const size_t, const float, const tart::buffer_ptr,
																					const size_t, const size_t, tart::buffer_ptr, const size_t);
template StatusCode PUBLIC_API Hpr<double>(const Layout, const Triangle, const size_t, const double, const tart::buffer_ptr,
																					 const size_t, const size_t, tart::buffer_ptr, const size_t);

// Hermitian rank-2 matrix update: CHER2/ZHER2
template <typename T>
StatusCode Her2(const Layout layout, const Triangle triangle, const size_t n, const T alpha, const tart::buffer_ptr x_buffer,
								const size_t x_offset, const size_t x_inc, const tart::buffer_ptr y_buffer, const size_t y_offset,
								const size_t y_inc, tart::buffer_ptr a_buffer, const size_t a_offset, const size_t a_ld) {
	try {
		auto device = x_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Her2"}, dtype);
		auto routine = Xher2(dtype, queue_cpp);
		routine.DoHer2(layout, triangle, n, alpha, x_buffer, x_offset, x_inc, y_buffer, y_offset,
									 y_inc, a_buffer, a_offset, a_ld, false, "HER2", db, device, dtype);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Her2<float2>(const Layout, const Triangle, const size_t, const float2, const tart::buffer_ptr,
	const size_t, const size_t, const tart::buffer_ptr, const size_t, const size_t,
	tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Her2<double2>(const Layout, const Triangle, const size_t, const double2, const tart::buffer_ptr,
	const size_t, const size_t, const tart::buffer_ptr, const size_t, const size_t,
	tart::buffer_ptr, const size_t, const size_t);

// Hermitian packed rank-2 matrix update: CHPR2/ZHPR2
template <typename T>
StatusCode Hpr2(const Layout layout, const Triangle triangle, const size_t n, const T alpha, const tart::buffer_ptr x_buffer,
								const size_t x_offset, const size_t x_inc, const tart::buffer_ptr y_buffer, const size_t y_offset,
								const size_t y_inc, tart::buffer_ptr ap_buffer, const size_t ap_offset) {
	try {
		auto device = ap_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Her2"}, dtype);
		auto routine = Xhpr2(dtype, queue_cpp);
		routine.DoHpr2(layout, triangle, n, alpha, x_buffer, x_offset, x_inc, y_buffer, y_offset,
									 y_inc, ap_buffer, ap_offset, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Hpr2<float2>(const Layout, const Triangle, const size_t, const float2, const tart::buffer_ptr,
	const size_t, const size_t, const tart::buffer_ptr, const size_t, const size_t,
	tart::buffer_ptr, const size_t);
template StatusCode PUBLIC_API Hpr2<double2>(const Layout, const Triangle, const size_t, const double2, const tart::buffer_ptr,
	const size_t, const size_t, const tart::buffer_ptr, const size_t, const size_t,
	tart::buffer_ptr, const size_t);

// Symmetric rank-1 matrix update: SSYR/DSYR/HSYR
template <typename T>
StatusCode Syr(const Layout layout, const Triangle triangle, const size_t n, const T alpha, const tart::buffer_ptr x_buffer,
							 const size_t x_offset, const size_t x_inc, tart::buffer_ptr a_buffer, const size_t a_offset, const size_t a_ld) {
	try {
		auto device = x_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Her"}, dtype);
		auto routine = Xsyr(dtype, queue_cpp);
		routine.DoSyr(layout, triangle, n, alpha, x_buffer, x_offset, x_inc, a_buffer, a_offset,
									a_ld, device, dtype, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Syr<float>(const Layout, const Triangle, const size_t, const float, const tart::buffer_ptr,
	const size_t, const size_t, tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Syr<double>(const Layout, const Triangle, const size_t, const double, const tart::buffer_ptr,
	const size_t, const size_t, tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Syr<half>(const Layout, const Triangle, const size_t, const half, const tart::buffer_ptr,
	const size_t, const size_t, tart::buffer_ptr, const size_t, const size_t);

// Symmetric packed rank-1 matrix update: SSPR/DSPR/HSPR
template <typename T>
StatusCode Spr(const Layout layout, const Triangle triangle, const size_t n, const T alpha, const tart::buffer_ptr x_buffer,
							 const size_t x_offset, const size_t x_inc, tart::buffer_ptr ap_buffer, const size_t ap_offset) {
	try {
		auto device = x_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Her"}, dtype);
		auto routine = Xspr(dtype, queue_cpp);
		routine.DoSpr(layout, triangle, n, alpha, x_buffer, x_offset, x_inc, ap_buffer, ap_offset,
									device, dtype, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Spr<float>(const Layout, const Triangle, const size_t, const float, const tart::buffer_ptr,
																					const size_t, const size_t, tart::buffer_ptr, const size_t);
template StatusCode PUBLIC_API Spr<double>(const Layout, const Triangle, const size_t, const double, const tart::buffer_ptr,
																					 const size_t, const size_t, tart::buffer_ptr, const size_t);
template StatusCode PUBLIC_API Spr<half>(const Layout, const Triangle, const size_t, const half, const tart::buffer_ptr,
																				 const size_t, const size_t, tart::buffer_ptr, const size_t);

// Symmetric rank-2 matrix update: SSYR2/DSYR2/HSYR2
template <typename T>
StatusCode Syr2(const Layout layout, const Triangle triangle, const size_t n, const T alpha, const tart::buffer_ptr x_buffer,
								const size_t x_offset, const size_t x_inc, const tart::buffer_ptr y_buffer, const size_t y_offset,
								const size_t y_inc, tart::buffer_ptr a_buffer, const size_t a_offset, const size_t a_ld) {
	try {
		auto device = x_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Her2"}, dtype);
		auto routine = Xsyr2(dtype, queue_cpp);
		routine.DoSyr2(layout, triangle, n, alpha, x_buffer, x_offset, x_inc, y_buffer, y_offset,
									 y_inc, a_buffer, a_offset, a_ld, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Syr2<float>(const Layout, const Triangle, const size_t, const float, const tart::buffer_ptr,
																					 const size_t, const size_t, const tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr,
																					 const size_t, const size_t);
template StatusCode PUBLIC_API Syr2<double>(const Layout, const Triangle, const size_t, const double, const tart::buffer_ptr,
																						const size_t, const size_t, const tart::buffer_ptr, const size_t, const size_t,
																						tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Syr2<half>(const Layout, const Triangle, const size_t, const half, const tart::buffer_ptr,
																					const size_t, const size_t, const tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr,
																					const size_t, const size_t);

// Symmetric packed rank-2 matrix update: SSPR2/DSPR2/HSPR2
template <typename T>
StatusCode Spr2(const Layout layout, const Triangle triangle, const size_t n, const T alpha, const tart::buffer_ptr x_buffer,
								const size_t x_offset, const size_t x_inc, const tart::buffer_ptr y_buffer, const size_t y_offset,
								const size_t y_inc, tart::buffer_ptr ap_buffer, const size_t ap_offset) {
	try {
		auto device = ap_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Her2"}, dtype);
		auto routine = Xspr2(dtype, queue_cpp);
		routine.DoSpr2(layout, triangle, n, alpha, x_buffer, x_offset, x_inc, y_buffer, y_offset,
									 y_inc, ap_buffer, ap_offset, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Spr2<float>(const Layout, const Triangle, const size_t, const float, const tart::buffer_ptr,
																					 const size_t, const size_t, const tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr,
																					 const size_t);
template StatusCode PUBLIC_API Spr2<double>(const Layout, const Triangle, const size_t, const double, const tart::buffer_ptr,
																						const size_t, const size_t, const tart::buffer_ptr, const size_t, const size_t,
																						tart::buffer_ptr, const size_t);
template StatusCode PUBLIC_API Spr2<half>(const Layout, const Triangle, const size_t, const half, const tart::buffer_ptr,
																					const size_t, const size_t, const tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr,
																					const size_t);

// =================================================================================================
// BLAS level-3 (matrix-matrix) routines
// =================================================================================================

// General matrix-matrix multiplication: SGEMM/DGEMM/CGEMM/ZGEMM/HGEMM
template <typename T>
StatusCode Gemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose, const size_t m,
								const size_t n, const size_t k, const T alpha, const tart::buffer_ptr a_buffer, const size_t a_offset,
								const size_t a_ld, const tart::buffer_ptr b_buffer, const size_t b_offset, const size_t b_ld, const T beta,
								tart::buffer_ptr c_buffer, const size_t c_offset, const size_t c_ld,
								tart::buffer_ptr temp_buffer) {
	try {
		auto device = a_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Xgemm", "GemmRoutine"}, dtype);
		auto routine = Xgemm(dtype, queue_cpp);
		const auto temp_buffer_provided = temp_buffer != nullptr;
		auto temp_buffer_cpp = temp_buffer_provided ? temp_buffer : nullptr;
		routine.DoGemm(layout, a_transpose, b_transpose, m, n, k, alpha, a_buffer, a_offset, a_ld,
									 b_buffer, b_offset, b_ld, beta, c_buffer, c_offset, c_ld, temp_buffer_cpp,
									 temp_buffer_provided, db, device, dtype);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Gemm<float>(const Layout, const Transpose, const Transpose, const size_t, const size_t,
																					 const size_t, const float, const tart::buffer_ptr, const size_t, const size_t,
																					 const tart::buffer_ptr, const size_t, const size_t, const float, tart::buffer_ptr, const size_t,
																					 const size_t, tart::buffer_ptr);
template StatusCode PUBLIC_API Gemm<double>(const Layout, const Transpose, const Transpose, const size_t, const size_t,
																						const size_t, const double, const tart::buffer_ptr, const size_t, const size_t,
																						const tart::buffer_ptr, const size_t, const size_t, const double, tart::buffer_ptr,
																						const size_t, const size_t, tart::buffer_ptr);
template StatusCode PUBLIC_API Gemm<float2>(const Layout, const Transpose, const Transpose, const size_t, const size_t,
																						const size_t, const float2, const tart::buffer_ptr, const size_t, const size_t,
																						const tart::buffer_ptr, const size_t, const size_t, const float2, tart::buffer_ptr,
																						const size_t, const size_t, tart::buffer_ptr);
template StatusCode PUBLIC_API Gemm<double2>(const Layout, const Transpose, const Transpose, const size_t, const size_t,
																						 const size_t, const double2, const tart::buffer_ptr, const size_t, const size_t,
																						 const tart::buffer_ptr, const size_t, const size_t, const double2, tart::buffer_ptr,
																						 const size_t, const size_t, tart::buffer_ptr);
template StatusCode PUBLIC_API Gemm<half>(const Layout, const Transpose, const Transpose, const size_t, const size_t,
																					const size_t, const half, const tart::buffer_ptr, const size_t, const size_t,
																					const tart::buffer_ptr, const size_t, const size_t, const half, tart::buffer_ptr, const size_t,
																					const size_t, tart::buffer_ptr);

// Symmetric matrix-matrix multiplication: SSYMM/DSYMM/CSYMM/ZSYMM/HSYMM
template <typename T>
StatusCode Symm(const Layout layout, const Side side, const Triangle triangle, const size_t m, const size_t n,
								const T alpha, const tart::buffer_ptr a_buffer, const size_t a_offset, const size_t a_ld, const tart::buffer_ptr b_buffer,
								const size_t b_offset, const size_t b_ld, const T beta, tart::buffer_ptr c_buffer, const size_t c_offset,
								const size_t c_ld) {
	try {
		auto device = a_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Gemm"}, dtype);
		auto routine = Xsymm(dtype, queue_cpp);
		routine.DoSymm(layout, side, triangle, m, n, alpha, a_buffer, a_offset, a_ld, b_buffer,
									 b_offset, b_ld, beta, c_buffer, c_offset, c_ld, device, dtype, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Symm<float>(const Layout, const Side, const Triangle, const size_t, const size_t,
																					 const float, const tart::buffer_ptr, const size_t, const size_t, const tart::buffer_ptr,
																					 const size_t, const size_t, const float, tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Symm<double>(const Layout, const Side, const Triangle, const size_t, const size_t,
																						const double, const tart::buffer_ptr, const size_t, const size_t, const tart::buffer_ptr,
																						const size_t, const size_t, const double, tart::buffer_ptr, const size_t,
																						const size_t);
template StatusCode PUBLIC_API Symm<float2>(const Layout, const Side, const Triangle, const size_t, const size_t,
																						const float2, const tart::buffer_ptr, const size_t, const size_t, const tart::buffer_ptr,
																						const size_t, const size_t, const float2, tart::buffer_ptr, const size_t,
																						const size_t);
template StatusCode PUBLIC_API Symm<double2>(const Layout, const Side, const Triangle, const size_t, const size_t,
																						 const double2, const tart::buffer_ptr, const size_t, const size_t, const tart::buffer_ptr,
																						 const size_t, const size_t, const double2, tart::buffer_ptr, const size_t,
																						 const size_t);
template StatusCode PUBLIC_API Symm<half>(const Layout, const Side, const Triangle, const size_t, const size_t,
																					const half, const tart::buffer_ptr, const size_t, const size_t, const tart::buffer_ptr,
																					const size_t, const size_t, const half, tart::buffer_ptr, const size_t, const size_t);

// Hermitian matrix-matrix multiplication: CHEMM/ZHEMM
template <typename T>
StatusCode Hemm(const Layout layout, const Side side, const Triangle triangle, const size_t m, const size_t n,
								const T alpha, const tart::buffer_ptr a_buffer, const size_t a_offset, const size_t a_ld, const tart::buffer_ptr b_buffer,
								const size_t b_offset, const size_t b_ld, const T beta, tart::buffer_ptr c_buffer, const size_t c_offset,
								const size_t c_ld) {
	try {
		auto device = a_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Gemm"}, dtype);
		auto routine = Xhemm(dtype, queue_cpp);
		routine.DoHemm(layout, side, triangle, m, n, alpha, a_buffer, a_offset, a_ld, b_buffer,
									 b_offset, b_ld, beta, c_buffer, c_offset, c_ld, device, dtype, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Hemm<float2>(const Layout, const Side, const Triangle, const size_t, const size_t,
																						const float2, const tart::buffer_ptr, const size_t, const size_t, const tart::buffer_ptr,
																						const size_t, const size_t, const float2, tart::buffer_ptr, const size_t,
																						const size_t);
template StatusCode PUBLIC_API Hemm<double2>(const Layout, const Side, const Triangle, const size_t, const size_t,
																						 const double2, const tart::buffer_ptr, const size_t, const size_t, const tart::buffer_ptr,
																						 const size_t, const size_t, const double2, tart::buffer_ptr, const size_t,
																						 const size_t);

// Rank-K update of a symmetric matrix: SSYRK/DSYRK/CSYRK/ZSYRK/HSYRK
template <typename T>
StatusCode Syrk(const Layout layout, const Triangle triangle, const Transpose a_transpose, const size_t n,
								const size_t k, const T alpha, const tart::buffer_ptr a_buffer, const size_t a_offset, const size_t a_ld,
								const T beta, tart::buffer_ptr c_buffer, const size_t c_offset, const size_t c_ld) {
	try {
		auto device = a_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Syrk"}, dtype);
		auto routine = Xsyrk(dtype, queue_cpp);
		routine.DoSyrk(layout, triangle, a_transpose, n, k, alpha, a_buffer, a_offset, a_ld, beta,
									 c_buffer, c_offset, c_ld, db, device, dtype);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Syrk<float>(const Layout, const Triangle, const Transpose, const size_t, const size_t,
																					 const float, const tart::buffer_ptr, const size_t, const size_t, const float, tart::buffer_ptr,
																					 const size_t, const size_t);
template StatusCode PUBLIC_API Syrk<double>(const Layout, const Triangle, const Transpose, const size_t, const size_t,
																						const double, const tart::buffer_ptr, const size_t, const size_t, const double,
																						tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Syrk<float2>(const Layout, const Triangle, const Transpose, const size_t, const size_t,
																						const float2, const tart::buffer_ptr, const size_t, const size_t, const float2,
																						tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Syrk<double2>(const Layout, const Triangle, const Transpose, const size_t, const size_t,
																						 const double2, const tart::buffer_ptr, const size_t, const size_t, const double2,
																						 tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Syrk<half>(const Layout, const Triangle, const Transpose, const size_t, const size_t,
																					const half, const tart::buffer_ptr, const size_t, const size_t, const half, tart::buffer_ptr,
																					const size_t, const size_t);

// Rank-K update of a hermitian matrix: CHERK/ZHERK
template <typename T>
StatusCode Herk(const Layout layout, const Triangle triangle, const Transpose a_transpose, const size_t n,
								const size_t k, const T alpha, const tart::buffer_ptr a_buffer, const size_t a_offset, const size_t a_ld,
								const T beta, tart::buffer_ptr c_buffer, const size_t c_offset, const size_t c_ld) {
	try {
		auto device = a_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Herk"}, dtype);
		auto routine = Xherk(dtype, queue_cpp);
		routine.DoHerk(layout, triangle, a_transpose, n, k, alpha, a_buffer, a_offset, a_ld, beta,
									 c_buffer, c_offset, c_ld, db, device, dtype);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Herk<float>(const Layout, const Triangle, const Transpose, const size_t, const size_t,
																					 const float, const tart::buffer_ptr, const size_t, const size_t, const float, tart::buffer_ptr,
																					 const size_t, const size_t);
template StatusCode PUBLIC_API Herk<double>(const Layout, const Triangle, const Transpose, const size_t, const size_t,
																						const double, const tart::buffer_ptr, const size_t, const size_t, const double,
																						tart::buffer_ptr, const size_t, const size_t);

// Rank-2K update of a symmetric matrix: SSYR2K/DSYR2K/CSYR2K/ZSYR2K/HSYR2K
template <typename T>
StatusCode Syr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose, const size_t n,
								 const size_t k, const T alpha, const tart::buffer_ptr a_buffer, const size_t a_offset, const size_t a_ld,
								 const tart::buffer_ptr b_buffer, const size_t b_offset, const size_t b_ld, const T beta, tart::buffer_ptr c_buffer,
								 const size_t c_offset, const size_t c_ld) {
	try {
		auto device = a_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Syrk"}, dtype);
		auto routine = Xsyr2k(dtype, queue_cpp);
		routine.DoSyr2k(layout, triangle, ab_transpose, n, k, alpha, a_buffer, a_offset, a_ld,
										(b_buffer), b_offset, b_ld, beta, c_buffer, c_offset, c_ld, db, device, dtype);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Syr2k<float>(const Layout, const Triangle, const Transpose, const size_t, const size_t,
																						const float, const tart::buffer_ptr, const size_t, const size_t, const tart::buffer_ptr,
																						const size_t, const size_t, const float, tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Syr2k<double>(const Layout, const Triangle, const Transpose, const size_t, const size_t,
																						 const double, const tart::buffer_ptr, const size_t, const size_t, const tart::buffer_ptr,
																						 const size_t, const size_t, const double, tart::buffer_ptr, const size_t,
																						 const size_t);
template StatusCode PUBLIC_API Syr2k<float2>(const Layout, const Triangle, const Transpose, const size_t, const size_t,
																						 const float2, const tart::buffer_ptr, const size_t, const size_t, const tart::buffer_ptr,
																						 const size_t, const size_t, const float2, tart::buffer_ptr, const size_t,
																						 const size_t);
template StatusCode PUBLIC_API Syr2k<double2>(const Layout, const Triangle, const Transpose, const size_t, const size_t,
																							const double2, const tart::buffer_ptr, const size_t, const size_t, const tart::buffer_ptr,
																							const size_t, const size_t, const double2, tart::buffer_ptr, const size_t,
																							const size_t);
template StatusCode PUBLIC_API Syr2k<half>(const Layout, const Triangle, const Transpose, const size_t, const size_t,
																					 const half, const tart::buffer_ptr, const size_t, const size_t, const tart::buffer_ptr,
																					 const size_t, const size_t, const half, tart::buffer_ptr, const size_t, const size_t);

// Rank-2K update of a hermitian matrix: CHER2K/ZHER2K
template <typename T, typename U>
StatusCode Her2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose, const size_t n,
								 const size_t k, const T alpha, const tart::buffer_ptr a_buffer, const size_t a_offset, const size_t a_ld,
								 const tart::buffer_ptr b_buffer, const size_t b_offset, const size_t b_ld, const U beta, tart::buffer_ptr c_buffer,
								 const size_t c_offset, const size_t c_ld) {
	try {
		auto device = a_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Herk"}, dtype);
		auto routine = Xher2k(dtype, queue_cpp);
		routine.DoHer2k(layout, triangle, ab_transpose, n, k, alpha, a_buffer, a_offset, a_ld,
										(b_buffer), b_offset, b_ld, beta, c_buffer, c_offset, c_ld,
										device, dtype, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Her2k<float2, float>(const Layout, const Triangle, const Transpose, const size_t,
																										const size_t, const float2, const tart::buffer_ptr, const size_t,
																										const size_t, const tart::buffer_ptr, const size_t, const size_t, const float,
																										tart::buffer_ptr, const size_t, const size_t);
template StatusCode PUBLIC_API Her2k<double2, double>(const Layout, const Triangle, const Transpose, const size_t,
																											const size_t, const double2, const tart::buffer_ptr, const size_t,
																											const size_t, const tart::buffer_ptr, const size_t, const size_t,
																											const double, tart::buffer_ptr, const size_t, const size_t);

// Triangular matrix-matrix multiplication: STRMM/DTRMM/CTRMM/ZTRMM/HTRMM
template <typename T>
StatusCode Trmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose,
								const Diagonal diagonal, const size_t m, const size_t n, const T alpha, const tart::buffer_ptr a_buffer,
								const size_t a_offset, const size_t a_ld, tart::buffer_ptr b_buffer, const size_t b_offset, const size_t b_ld) {
	try {
		auto device = a_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Gemm"}, dtype);
		auto routine = Xtrmm(dtype, queue_cpp);
		routine.DoTrmm(layout, side, triangle, a_transpose, diagonal, m, n, alpha, a_buffer, a_offset, a_ld,
									 b_buffer, b_offset, b_ld,
									 device, dtype, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Trmm<float>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
																					 const size_t, const size_t, const float, const tart::buffer_ptr, const size_t,
																					 const size_t, tart::buffer_ptr, const size_t, const size_t
																					event);
template StatusCode PUBLIC_API Trmm<double>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
																						const size_t, const size_t, const double, const tart::buffer_ptr, const size_t,
																						const size_t, tart::buffer_ptr, const size_t, const size_t
																					event);
template StatusCode PUBLIC_API Trmm<float2>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
																						const size_t, const size_t, const float2, const tart::buffer_ptr, const size_t,
																						const size_t, tart::buffer_ptr, const size_t, const size_t
																					event);
template StatusCode PUBLIC_API Trmm<double2>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
																						 const size_t, const size_t, const double2, const tart::buffer_ptr, const size_t,
																						 const size_t, tart::buffer_ptr, const size_t, const size_t
																					event);
template StatusCode PUBLIC_API Trmm<half>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
																					const size_t, const size_t, const half, const tart::buffer_ptr, const size_t,
																					const size_t, tart::buffer_ptr, const size_t, const size_t
																					event);

// Solves a triangular system of equations: STRSM/DTRSM/CTRSM/ZTRSM
template <typename T>
StatusCode Trsm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose,
								const Diagonal diagonal, const size_t m, const size_t n, const T alpha, const tart::buffer_ptr a_buffer,
								const size_t a_offset, const size_t a_ld, tart::buffer_ptr b_buffer, const size_t b_offset, const size_t b_ld) {
	try {
		auto device = a_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Gemm", "Trsm"}, dtype);
		auto routine = Xtrsm(dtype, queue_cpp);
		routine.DoTrsm(layout, side, triangle, a_transpose, diagonal, m, n, alpha, a_buffer, a_offset, a_ld,
									 b_buffer, b_offset, b_ld, device, dtype, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Trsm<float>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
																					 const size_t, const size_t, const float, const tart::buffer_ptr, const size_t,
																					 const size_t, tart::buffer_ptr, const size_t, const size_t
																					event);
template StatusCode PUBLIC_API Trsm<double>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
																						const size_t, const size_t, const double, const tart::buffer_ptr, const size_t,
																						const size_t, tart::buffer_ptr, const size_t, const size_t
																					event);
template StatusCode PUBLIC_API Trsm<float2>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
																						const size_t, const size_t, const float2, const tart::buffer_ptr, const size_t,
																						const size_t, tart::buffer_ptr, const size_t, const size_t
																					event);
template StatusCode PUBLIC_API Trsm<double2>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
																						 const size_t, const size_t, const double2, const tart::buffer_ptr, const size_t,
																						 const size_t, tart::buffer_ptr, const size_t, const size_t
																					event);

// =================================================================================================
// Extra non-BLAS routines (level-X)
// =================================================================================================

// Element-wise vector product (Hadamard): SHAD/DHAD/CHAD/ZHAD/HHAD
template <typename T>
StatusCode Had(const size_t n, const T alpha, const tart::buffer_ptr x_buffer, const size_t x_offset, const size_t x_inc,
							 const tart::buffer_ptr y_buffer, const size_t y_offset, const size_t y_inc, const T beta, tart::buffer_ptr z_buffer,
							 const size_t z_offset, const size_t z_inc) {
	try {
		auto device = x_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Xhad"}, dtype);
		DoHad(n, alpha, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, beta,
					z_buffer, z_offset, z_inc, device, dtype, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Had<float>(const size_t, const float, const tart::buffer_ptr, const size_t, const size_t,
																					const tart::buffer_ptr, const size_t, const size_t, const float, tart::buffer_ptr, const size_t,
																					const size_t);
template StatusCode PUBLIC_API Had<double>(const size_t, const double, const tart::buffer_ptr, const size_t, const size_t,
																					 const tart::buffer_ptr, const size_t, const size_t, const double, tart::buffer_ptr, const size_t,
																					 const size_t);
template StatusCode PUBLIC_API Had<float2>(const size_t, const float2, const tart::buffer_ptr, const size_t, const size_t,
																					 const tart::buffer_ptr, const size_t, const size_t, const float2, tart::buffer_ptr, const size_t,
																					 const size_t);
template StatusCode PUBLIC_API Had<double2>(const size_t, const double2, const tart::buffer_ptr, const size_t, const size_t,
																						const tart::buffer_ptr, const size_t, const size_t, const double2, tart::buffer_ptr,
																						const size_t, const size_t);
template StatusCode PUBLIC_API Had<half>(const size_t, const half, const tart::buffer_ptr, const size_t, const size_t,
																				 const tart::buffer_ptr, const size_t, const size_t, const half, tart::buffer_ptr, const size_t,
																				 const size_t);

// Scaling and out-place transpose/copy (non-BLAS function): SOMATCOPY/DOMATCOPY/COMATCOPY/ZOMATCOPY/HOMATCOPY
template <typename T>
StatusCode Omatcopy(const Layout layout, const Transpose a_transpose, const size_t m, const size_t n, const T alpha,
										const tart::buffer_ptr a_buffer, const size_t a_offset, const size_t a_ld, tart::buffer_ptr b_buffer,
										const size_t b_offset, const size_t b_ld) {
	try {
		auto device = a_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Omatcopy"}, dtype);
		DoOmatcopy(layout, a_transpose, m, n, alpha, a_buffer, a_offset, a_ld, b_buffer,
										 b_offset, b_ld, db, device, dtype);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Omatcopy<float>(const Layout, const Transpose, const size_t, const size_t, const float,
																							 const tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr, const size_t,
																							 const size_t);
template StatusCode PUBLIC_API Omatcopy<double>(const Layout, const Transpose, const size_t, const size_t, const double,
																								const tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr, const size_t,
																								const size_t);
template StatusCode PUBLIC_API Omatcopy<float2>(const Layout, const Transpose, const size_t, const size_t, const float2,
																								const tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr, const size_t,
																								const size_t);
template StatusCode PUBLIC_API Omatcopy<double2>(const Layout, const Transpose, const size_t, const size_t,
																								 const double2, const tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr,
																								 const size_t, const size_t);
template StatusCode PUBLIC_API Omatcopy<half>(const Layout, const Transpose, const size_t, const size_t, const half,
																							const tart::buffer_ptr, const size_t, const size_t, tart::buffer_ptr, const size_t,
																							const size_t);

// Im2col function (non-BLAS function): SIM2COL/DIM2COL/CIM2COL/ZIM2COL/HIM2COL
template <typename T>
StatusCode Im2col(const KernelMode kernel_mode, const size_t channels, const size_t height, const size_t width,
									const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w,
									const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
									const tart::buffer_ptr im_buffer, const size_t im_offset, tart::buffer_ptr col_buffer, const size_t col_offset) {
	try {
		auto device = im_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Im2col"}, dtype);
		DoIm2col(kernel_mode, channels, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
						 dilation_h, dilation_w, im_buffer, im_offset, col_buffer, col_offset, device, dtype, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Im2col<float>(const KernelMode, const size_t, const size_t, const size_t, const size_t,
																						 const size_t, const size_t, const size_t, const size_t, const size_t,
																						 const size_t, const size_t, const tart::buffer_ptr, const size_t, tart::buffer_ptr,
																						 const size_t);
template StatusCode PUBLIC_API Im2col<double>(const KernelMode, const size_t, const size_t, const size_t, const size_t,
																							const size_t, const size_t, const size_t, const size_t, const size_t,
																							const size_t, const size_t, const tart::buffer_ptr, const size_t, tart::buffer_ptr,
																							const size_t);
template StatusCode PUBLIC_API Im2col<float2>(const KernelMode, const size_t, const size_t, const size_t, const size_t,
																							const size_t, const size_t, const size_t, const size_t, const size_t,
																							const size_t, const size_t, const tart::buffer_ptr, const size_t, tart::buffer_ptr,
																							const size_t);
template StatusCode PUBLIC_API Im2col<double2>(const KernelMode, const size_t, const size_t, const size_t, const size_t,
																							 const size_t, const size_t, const size_t, const size_t, const size_t,
																							 const size_t, const size_t, const tart::buffer_ptr, const size_t, tart::buffer_ptr,
																							 const size_t);
template StatusCode PUBLIC_API Im2col<half>(const KernelMode, const size_t, const size_t, const size_t, const size_t,
																						const size_t, const size_t, const size_t, const size_t, const size_t,
																						const size_t, const size_t, const tart::buffer_ptr, const size_t, tart::buffer_ptr,
																						const size_t);

// Col2im function (non-BLAS function): SCOL2IM/DCOL2IM/CCOL2IM/ZCOL2IM/HCOL2IM
template <typename T>
StatusCode Col2im(const KernelMode kernel_mode, const size_t channels, const size_t height, const size_t width,
									const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w,
									const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
									const tart::buffer_ptr col_buffer, const size_t col_offset, tart::buffer_ptr im_buffer, const size_t im_offset) {
	try {
		auto device = col_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Col2im"}, dtype);
		DoCol2im(kernel_mode, channels, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
						 dilation_h, dilation_w, col_buffer, col_offset, im_buffer, im_offset, device, dtype, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Col2im<float>(const KernelMode, const size_t, const size_t, const size_t, const size_t,
																						 const size_t, const size_t, const size_t, const size_t, const size_t,
																						 const size_t, const size_t, const tart::buffer_ptr, const size_t, tart::buffer_ptr,
																						 const size_t);
template StatusCode PUBLIC_API Col2im<double>(const KernelMode, const size_t, const size_t, const size_t, const size_t,
																							const size_t, const size_t, const size_t, const size_t, const size_t,
																							const size_t, const size_t, const tart::buffer_ptr, const size_t, tart::buffer_ptr,
																							const size_t);
template StatusCode PUBLIC_API Col2im<float2>(const KernelMode, const size_t, const size_t, const size_t, const size_t,
																							const size_t, const size_t, const size_t, const size_t, const size_t,
																							const size_t, const size_t, const tart::buffer_ptr, const size_t, tart::buffer_ptr,
																							const size_t);
template StatusCode PUBLIC_API Col2im<double2>(const KernelMode, const size_t, const size_t, const size_t, const size_t,
																							 const size_t, const size_t, const size_t, const size_t, const size_t,
																							 const size_t, const size_t, const tart::buffer_ptr, const size_t, tart::buffer_ptr,
																							 const size_t);
template StatusCode PUBLIC_API Col2im<half>(const KernelMode, const size_t, const size_t, const size_t, const size_t,
																						const size_t, const size_t, const size_t, const size_t, const size_t,
																						const size_t, const size_t, const tart::buffer_ptr, const size_t, tart::buffer_ptr,
																						const size_t);

// Batched convolution as GEMM (non-BLAS function): SCONVGEMM/DCONVGEMM/HCONVGEMM
template <typename T>
StatusCode Convgemm(const KernelMode kernel_mode, const size_t channels, const size_t height, const size_t width,
										const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w,
										const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
										const size_t num_kernels, const size_t batch_count, const tart::buffer_ptr im_buffer, const size_t im_offset,
										const tart::buffer_ptr kernel_buffer, const size_t kernel_offset, tart::buffer_ptr result_buffer,
										const size_t result_offset) {
	try {
		auto device = im_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"Convgemm"}, dtype);
		DoConvgemm(kernel_mode, channels, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
											 dilation_h, dilation_w, num_kernels, batch_count, im_buffer, im_offset,
											 kernel_buffer, kernel_offset, result_buffer, result_offset, "CONVGEMM", device, dtype, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API Convgemm<float>(const KernelMode, const size_t, const size_t, const size_t, const size_t,
																							 const size_t, const size_t, const size_t, const size_t, const size_t,
																							 const size_t, const size_t, const size_t, const size_t, const tart::buffer_ptr,
																							 const size_t, const tart::buffer_ptr, const size_t, tart::buffer_ptr, const size_t);
template StatusCode PUBLIC_API Convgemm<double>(const KernelMode, const size_t, const size_t, const size_t,
																								const size_t, const size_t, const size_t, const size_t, const size_t,
																								const size_t, const size_t, const size_t, const size_t, const size_t,
																								const tart::buffer_ptr, const size_t, const tart::buffer_ptr, const size_t, tart::buffer_ptr,
																								const size_t);
template StatusCode PUBLIC_API Convgemm<half>(const KernelMode, const size_t, const size_t, const size_t, const size_t,
																							const size_t, const size_t, const size_t, const size_t, const size_t,
																							const size_t, const size_t, const size_t, const size_t, const tart::buffer_ptr,
																							const size_t, const tart::buffer_ptr, const size_t, tart::buffer_ptr, const size_t);

// Batched version of AXPY: SAXPYBATCHED/DAXPYBATCHED/CAXPYBATCHED/ZAXPYBATCHED/HAXPYBATCHED
template <typename T>
StatusCode AxpyBatched(const size_t n, const T* alphas, const tart::buffer_ptr x_buffer, const size_t* x_offsets,
											 const size_t x_inc, tart::buffer_ptr y_buffer, const size_t* y_offsets, const size_t y_inc,
											 const size_t batch_count) {
	try {
		auto device = x_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"XaxpyBatched"}, dtype);
		auto alphas_cpp = std::vector<T>();
		auto x_offsets_cpp = std::vector<size_t>();
		auto y_offsets_cpp = std::vector<size_t>();
		for (auto batch = size_t{0}; batch < batch_count; ++batch) {
			alphas_cpp.push_back(alphas[batch]);
			x_offsets_cpp.push_back(x_offsets[batch]);
			y_offsets_cpp.push_back(y_offsets[batch]);
		}
		auto alphas_device = device->allocateBuffer(batch_count * sizeof(T));
		alphas_device->copyIn(alphas_cpp);
		DoAxpyBatched(n, alphas_device, x_buffer, x_offsets_cpp, x_inc, y_buffer, y_offsets_cpp,
										y_inc, batch_count, device, dtype, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API AxpyBatched<float>(const size_t, const float*, const tart::buffer_ptr, const size_t*, const size_t,
																									tart::buffer_ptr, const size_t*, const size_t, const size_t);
template StatusCode PUBLIC_API AxpyBatched<double>(const size_t, const double*, const tart::buffer_ptr, const size_t*,
																									 const size_t, tart::buffer_ptr, const size_t*, const size_t, const size_t);
template StatusCode PUBLIC_API AxpyBatched<float2>(const size_t, const float2*, const tart::buffer_ptr, const size_t*,
																									 const size_t, tart::buffer_ptr, const size_t*, const size_t, const size_t);
template StatusCode PUBLIC_API AxpyBatched<double2>(const size_t, const double2*, const tart::buffer_ptr, const size_t*,
																										const size_t, tart::buffer_ptr, const size_t*, const size_t, const size_t);
template StatusCode PUBLIC_API AxpyBatched<half>(const size_t, const half*, const tart::buffer_ptr, const size_t*, const size_t,
																								 tart::buffer_ptr, const size_t*, const size_t, const size_t);

// Batched version of GEMM: SGEMMBATCHED/DGEMMBATCHED/CGEMMBATCHED/ZGEMMBATCHED/HGEMMBATCHED
template <typename T>
StatusCode GemmBatched(const Layout layout, const Transpose a_transpose, const Transpose b_transpose, const size_t m,
											 const size_t n, const size_t k, const T* alphas, const tart::buffer_ptr a_buffer, const size_t* a_offsets,
											 const size_t a_ld, const tart::buffer_ptr b_buffer, const size_t* b_offsets, const size_t b_ld,
											 const T* betas, tart::buffer_ptr c_buffer, const size_t* c_offsets, const size_t c_ld,
											 const size_t batch_count) {
	try {
		auto device = a_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"XgemmBatched"}, dtype);
		auto alphas_cpp = std::vector<T>();
		auto betas_cpp = std::vector<T>();
		auto a_offsets_cpp = std::vector<size_t>();
		auto b_offsets_cpp = std::vector<size_t>();
		auto c_offsets_cpp = std::vector<size_t>();
		for (auto batch = size_t{0}; batch < batch_count; ++batch) {
			alphas_cpp.push_back(alphas[batch]);
			betas_cpp.push_back(betas[batch]);
			a_offsets_cpp.push_back(a_offsets[batch]);
			b_offsets_cpp.push_back(b_offsets[batch]);
			c_offsets_cpp.push_back(c_offsets[batch]);
		}
		auto alphas_device = device->allocateBuffer(batch_count * sizeof(T));
		auto betas_device = device->allocateBuffer(batch_count * sizeof(T));
		alphas_device->copyIn(alphas_cpp);
		betas_device->copyIn(betas_cpp);
		DoGemmBatched(layout, a_transpose, b_transpose, m, n, k, alphas_device, a_buffer, a_offsets_cpp,
										a_ld, b_buffer, b_offsets_cpp, b_ld, betas_device, c_buffer, c_offsets_cpp,
										c_ld, batch_count, device, dtype, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API GemmBatched<float>(const Layout, const Transpose, const Transpose, const size_t,
																									const size_t, const size_t, const float*, const tart::buffer_ptr, const size_t*,
																									const size_t, const tart::buffer_ptr, const size_t*, const size_t, const float*,
																									tart::buffer_ptr, const size_t*, const size_t, const size_t);
template StatusCode PUBLIC_API GemmBatched<double>(const Layout, const Transpose, const Transpose, const size_t,
																									 const size_t, const size_t, const double*, const tart::buffer_ptr,
																									 const size_t*, const size_t, const tart::buffer_ptr, const size_t*,
																									 const size_t, const double*, tart::buffer_ptr, const size_t*, const size_t,
																									 const size_t);
template StatusCode PUBLIC_API GemmBatched<float2>(const Layout, const Transpose, const Transpose, const size_t,
																									 const size_t, const size_t, const float2*, const tart::buffer_ptr,
																									 const size_t*, const size_t, const tart::buffer_ptr, const size_t*,
																									 const size_t, const float2*, tart::buffer_ptr, const size_t*, const size_t,
																									 const size_t);
template StatusCode PUBLIC_API GemmBatched<double2>(const Layout, const Transpose, const Transpose, const size_t,
																										const size_t, const size_t, const double2*, const tart::buffer_ptr,
																										const size_t*, const size_t, const tart::buffer_ptr, const size_t*,
																										const size_t, const double2*, tart::buffer_ptr, const size_t*, const size_t,
																										const size_t);
template StatusCode PUBLIC_API GemmBatched<half>(const Layout, const Transpose, const Transpose, const size_t,
																								 const size_t, const size_t, const half*, const tart::buffer_ptr, const size_t*,
																								 const size_t, const tart::buffer_ptr, const size_t*, const size_t, const half*,
																								 tart::buffer_ptr, const size_t*, const size_t, const size_t);

// StridedBatched version of GEMM:
// SGEMMSTRIDEDBATCHED/DGEMMSTRIDEDBATCHED/CGEMMSTRIDEDBATCHED/ZGEMMSTRIDEDBATCHED/HGEMMSTRIDEDBATCHED
template <typename T>
StatusCode GemmStridedBatched(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
															const size_t m, const size_t n, const size_t k, const T alpha, const tart::buffer_ptr a_buffer,
															const size_t a_offset, const size_t a_ld, const size_t a_stride, const tart::buffer_ptr b_buffer,
															const size_t b_offset, const size_t b_ld, const size_t b_stride, const T beta,
															tart::buffer_ptr c_buffer, const size_t c_offset, const size_t c_ld, const size_t c_stride,
															const size_t batch_count) {
	try {
		auto device = a_buffer->getDevice();
		auto queue_cpp = Queue(device);
		auto dtype = tart::getDType<T>();
		auto db = GetDatabases(device, {"XgemmStridedBatched"}, dtype);
		DoGemmStridedBatched(layout, a_transpose, b_transpose, m, n, k, alpha, a_buffer, a_offset, a_ld,
														 a_stride, b_buffer, b_offset, b_ld, b_stride, beta, c_buffer,
														 c_offset, c_ld, c_stride, batch_count, device, dtype, db);
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API GemmStridedBatched<float>(const Layout, const Transpose, const Transpose, const size_t,
																												 const size_t, const size_t, const float, const tart::buffer_ptr,
																												 const size_t, const size_t, const size_t, const tart::buffer_ptr,
																												 const size_t, const size_t, const size_t, const float, tart::buffer_ptr,
																												 const size_t, const size_t, const size_t, const size_t);
template StatusCode PUBLIC_API GemmStridedBatched<double>(const Layout, const Transpose, const Transpose, const size_t,
																													const size_t, const size_t, const double, const tart::buffer_ptr,
																													const size_t, const size_t, const size_t, const tart::buffer_ptr,
																													const size_t, const size_t, const size_t, const double,
																													tart::buffer_ptr, const size_t, const size_t, const size_t,
																													const size_t);
template StatusCode PUBLIC_API GemmStridedBatched<float2>(const Layout, const Transpose, const Transpose, const size_t,
																													const size_t, const size_t, const float2, const tart::buffer_ptr,
																													const size_t, const size_t, const size_t, const tart::buffer_ptr,
																													const size_t, const size_t, const size_t, const float2,
																													tart::buffer_ptr, const size_t, const size_t, const size_t,
																													const size_t);
template StatusCode PUBLIC_API GemmStridedBatched<double2>(const Layout, const Transpose, const Transpose, const size_t,
																													 const size_t, const size_t, const double2, const tart::buffer_ptr,
																													 const size_t, const size_t, const size_t, const tart::buffer_ptr,
																													 const size_t, const size_t, const size_t, const double2,
																													 tart::buffer_ptr, const size_t, const size_t, const size_t,
																													 const size_t);
template StatusCode PUBLIC_API GemmStridedBatched<half>(const Layout, const Transpose, const Transpose, const size_t,
																												const size_t, const size_t, const half, const tart::buffer_ptr,
																												const size_t, const size_t, const size_t, const tart::buffer_ptr,
																												const size_t, const size_t, const size_t, const half, tart::buffer_ptr,
																												const size_t, const size_t, const size_t, const size_t);

// =================================================================================================

// Retrieves the required size of the temporary buffer for the GEMM kernel (optional)
template <typename T>
StatusCode GemmTempBufferSize(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
															const size_t m, const size_t n, const size_t k, const size_t a_offset, const size_t a_ld,
															const size_t b_offset, const size_t b_ld, const size_t c_offset, const size_t c_ld,
															tart::device_ptr device, size_t& temp_buffer_size) {
	try {
		const auto kernel_names = std::vector<std::string>{"Xgemm", "GemmRoutine"};
		const auto dtype = tart::getDType<T>();
		const auto db = GetDatabases(device, kernel_names, dtype);

		// Computes the buffer size
		if (UseDirectKernel(m, n, k, db["XGEMM_MIN_INDIRECT_SIZE"])) {
			temp_buffer_size = 0;
		} else {
			temp_buffer_size =
					GetTempSize(layout, a_transpose, b_transpose, m, n, k, a_offset, a_ld, b_offset, b_ld, c_offset,
															c_ld, db["MWG"], db["NWG"], db["KWG"] * db["KREG"], db["GEMMK"]);
		}
		temp_buffer_size *= sizeof(T);	// translate from num-elements to bytes
		return StatusCode::kSuccess;
	} catch (...) {
		return DispatchException();
	}
}
template StatusCode PUBLIC_API GemmTempBufferSize<float>(const Layout, const Transpose, const Transpose, const size_t,
																												 const size_t, const size_t, const size_t, const size_t,
																												 const size_t, const size_t, const size_t, const size_t,
																												 tart::device_ptr, size_t&);
template StatusCode PUBLIC_API GemmTempBufferSize<double>(const Layout, const Transpose, const Transpose, const size_t,
																													const size_t, const size_t, const size_t, const size_t,
																													const size_t, const size_t, const size_t, const size_t,
																													tart::device_ptr, size_t&);
template StatusCode PUBLIC_API GemmTempBufferSize<float2>(const Layout, const Transpose, const Transpose, const size_t,
																													const size_t, const size_t, const size_t, const size_t,
																													const size_t, const size_t, const size_t, const size_t,
																													tart::device_ptr, size_t&);
template StatusCode PUBLIC_API GemmTempBufferSize<double2>(const Layout, const Transpose, const Transpose, const size_t,
																													 const size_t, const size_t, const size_t, const size_t,
																													 const size_t, const size_t, const size_t, const size_t,
																													 tart::device_ptr, size_t&);
template StatusCode PUBLIC_API GemmTempBufferSize<half>(const Layout, const Transpose, const Transpose, const size_t,
																												const size_t, const size_t, const size_t, const size_t,
																												const size_t, const size_t, const size_t, const size_t,
																												tart::device_ptr, size_t&);

// =================================================================================================
}	// namespace clblast
