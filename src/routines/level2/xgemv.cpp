
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the functional Xgemv routine.
//
// =================================================================================================

#include "routines/level2/xgemv.hpp"

#include <cstddef>
#include <string>
#include <vector>

#include "database/database.hpp"
#include "routine_aot_cache.hpp"
#include "routines/common.hpp"
#include "utilities/backend.hpp"
#include "utilities/buffer_test.hpp"
#include "utilities/clblast_exceptions.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// The main routine
void DoGemv(const Layout layout, const Transpose a_transpose, const size_t m, const size_t n,
            const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld,
            const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
            const tart::Scalar& beta, const tart::buffer_ptr& y_buffer, const size_t y_offset, const size_t y_inc,
            const Databases& db,
            tart::device_ptr device, const tart::DType& dtype) {
	// Performs the matrix-vector multiplication
	MatVec(layout, a_transpose, m, n, alpha, a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, beta, y_buffer,
				 y_offset, y_inc, true, true, 0, false, 0, 0, "GEMV", db, device, dtype);
}

// =================================================================================================

// The generic implementation, also suited for other (non general) matrix-vector multiplications
void MatVec(const Layout layout, const Transpose a_transpose, const size_t m, const size_t n,
            const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld,
            const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
            const tart::Scalar& beta, const tart::buffer_ptr& y_buffer, const size_t y_offset, const size_t y_inc,
            bool fast_kernel, bool fast_kernel_rot, const size_t parameter, const bool packed,
            const size_t kl, const size_t ku, const std::string& name, const Databases& db,
            tart::device_ptr device, const tart::DType& dtype) {
	// Makes sure all dimensions are larger than zero
	if (m == 0 || n == 0) {
		throw BLASError(StatusCode::kInvalidDimension);
	}

	// Computes whether or not the matrix has an alternative layout (row or column-major).
	const auto a_altlayout = (layout == Layout::kRowMajor);
	auto a_one = (a_altlayout) ? n : m;
	const auto a_two = (a_altlayout) ? m : n;

	// Swap m and n if the matrix is transposed
	const auto a_transposed = (a_transpose != Transpose::kNo);
	const auto m_real = (a_transposed) ? n : m;
	const auto n_real = (a_transposed) ? m : n;

	// Special adjustments for banded matrices
	if (kl != 0 || ku != 0) {
		a_one = kl + ku + 1;
	}

	// Determines whether the kernel needs to perform rotated access ('^' is the XOR operator)
	const auto a_rotated = a_transposed ^ a_altlayout;

	// In case of complex data-types, the transpose can also become a conjugate transpose
	const auto a_conjugate = (a_transpose == Transpose::kConjugate);

	// Tests the matrix and the vectors for validity
	if (packed) {
		TestMatrixAP(dtype, n, a_buffer, a_offset);
	} else {
		TestMatrixA(dtype, a_one, a_two, a_buffer, a_offset, a_ld);
	}
	TestVectorX(dtype, n_real, x_buffer, x_offset, x_inc);
	TestVectorY(dtype, m_real, y_buffer, y_offset, y_inc);

	// Determines whether or not the fast-version can be used
	fast_kernel = fast_kernel && (a_offset == 0) && (a_rotated == 0) && (a_conjugate == 0) &&
								IsMultiple(m, db["WGS2"] * db["WPT2"]) && IsMultiple(n, db["WGS2"]) && IsMultiple(a_ld, db["VW2"]);
	fast_kernel_rot = fast_kernel_rot && (a_offset == 0) && (a_rotated == 1) && (a_conjugate == 0) &&
										IsMultiple(m, db["WGS3"] * db["WPT3"]) && IsMultiple(n, db["WGS3"]) &&
										IsMultiple(a_ld, db["VW3"]);

	// Compute the flags from name (same as the old constructor)
	const auto mGBMV = (name == "GBMV");
	const auto mHEMV = (name == "HEMV");
	const auto mSYMV = (name == "SYMV");
	const auto mTRMV = (name == "TRMV");
	const auto mHBMV = (name == "HBMV");
	const auto mSBMV = (name == "SBMV");
	const auto mTBMV = (name == "TBMV");
	const auto mHPMV = (name == "HPMV");
	const auto mSPMV = (name == "SPMV");
	const auto mTPMV = (name == "TPMV");

	// If possible, run the fast-version (rotated or non-rotated) of the kernel
	auto kernel_name = std::string{"Xgemv"};
	const auto m_ceiled = Ceil(m_real, db["WGS1"] * db["WPT1"]);
	auto global_size = m_ceiled / db["WPT1"];
	auto local_size = db["WGS1"];
	tart::kernel_ptr kernel = nullptr;
	if (fast_kernel)
	{
		kernel = getGemvFast(device, dtype, db["VW2"]);
		global_size = m_real / db["WPT2"];
		local_size = db["WGS2"];
	}
	else if (fast_kernel_rot)
	{
		kernel = getGemvFastRot(device, dtype, db["VW3"]);
		global_size = m_real;
		local_size = db["WGS3"];
	}
	else
	{
		kernel = getGemv(device, dtype);
	}
	
#if VULKAN_API
	if (fast_kernel || fast_kernel_rot)
	{
		// Sets the kernel arguments
		// some are unused if fast_kernel or fast_kernel_rot is true
		kernel->setArg(0, static_cast<int>(m_real));
		kernel->setArg(1, static_cast<int>(n_real));
		kernel->setArgAs(dtype, 2, alpha);
		kernel->setArgAs(dtype, 3, beta);
		//kernel->setArg(4, static_cast<int>(a_rotated));
		kernel->setArg(5, a_buffer);
		//kernel->setArg(6, static_cast<int>(a_offset));
		kernel->setArg(7, static_cast<int>(a_ld));
		kernel->setArg(8, x_buffer);
		kernel->setArg(9, static_cast<int>(x_offset));
		kernel->setArg(10, static_cast<int>(x_inc));
		kernel->setArg(11, y_buffer);
		kernel->setArg(12, static_cast<int>(y_offset));
		kernel->setArg(13, static_cast<int>(y_inc));
		//kernel->setArg(14, static_cast<int>(a_conjugate));
		//kernel->setArg(15, static_cast<int>(parameter));	// extra parameter used for symm/herm
		//kernel->setArg(16, static_cast<int>(kl));				 // only used for banded matrices
		//kernel->setArg(17, static_cast<int>(ku));				 // only used for banded matrices
	}
	else
#endif
	{
		// Sets the kernel arguments
		kernel->setArg(0, static_cast<int>(m_real));
		kernel->setArg(1, static_cast<int>(n_real));
		kernel->setArgAs(dtype, 2, alpha);
		kernel->setArgAs(dtype, 3, beta);
		kernel->setArg(4, static_cast<int>(a_rotated));
		kernel->setArg(5, a_buffer);
		kernel->setArg(6, static_cast<int>(a_offset));
		kernel->setArg(7, static_cast<int>(a_ld));
		kernel->setArg(8, x_buffer);
		kernel->setArg(9, static_cast<int>(x_offset));
		kernel->setArg(10, static_cast<int>(x_inc));
		kernel->setArg(11, y_buffer);
		kernel->setArg(12, static_cast<int>(y_offset));
		kernel->setArg(13, static_cast<int>(y_inc));
		kernel->setArg(14, static_cast<int>(a_conjugate));
		kernel->setArg(15, static_cast<int>(parameter));	// extra parameter used for symm/herm
		kernel->setArg(16, static_cast<int>(kl));				 // only used for banded matrices
		kernel->setArg(17, static_cast<int>(ku));				 // only used for banded matrices
	}

	// Launches the kernel
	auto global = std::vector<uint32_t>{global_size / local_size};
	//auto local = std::vector<size_t>{local_size};
	
	std::vector<uint32_t> spec;
	if (fast_kernel)
	{
		spec = {
			db["WGS2"],
			db["WPT2"]
		};
	}
	else if (fast_kernel_rot)
	{
		spec = {
			db["WGS3"],
			db["WPT3"]
		};
	}
	else
	{
		spec = {
			db["WGS1"],
			db["WPT1"],
			32, //db["UNROLL1"], // apparently this isn't in the database for whatever reason
			mGBMV,
			mHEMV,
			mSYMV,
			mTRMV,
			mHBMV,
			mSBMV,
			mTBMV,
			mHPMV,
			mSPMV,
			mTPMV
		};
	}
	kernel->enqueue(global, spec);
}

// =================================================================================================
}	// namespace clblast
