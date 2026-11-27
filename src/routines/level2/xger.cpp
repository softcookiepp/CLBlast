
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the functional Xger routine.
//
// =================================================================================================

#include "routines/level2/xger.hpp"

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
void DoGer(const Layout layout, const uint32_t m, const uint32_t n, const tart::Scalar& alpha,
           const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
           const tart::buffer_ptr& y_buffer, const size_t y_offset, const size_t y_inc,
           const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld,
           const std::string& name, [[maybe_unused]] const Databases& db,
           tart::device_ptr device, const tart::DType& dtype) {
	// Makes sure all dimensions are larger than zero
	if (m == 0 || n == 0) {
		throw BLASError(StatusCode::kInvalidDimension);
	}

	// Computes whether or not the matrix has an alternative layout (row or column-major).
	const auto a_is_rowmajor = (layout == Layout::kRowMajor);
	const auto a_one = (a_is_rowmajor) ? n : m;
	const auto a_two = (a_is_rowmajor) ? m : n;

	// Tests the matrix and the vectors for validity
	TestMatrixA(dtype, a_one, a_two, a_buffer, a_offset, a_ld);
	TestVectorX(dtype, m, x_buffer, x_offset, x_inc);
	TestVectorY(dtype, n, y_buffer, y_offset, y_inc);

	// Compute the flags from name
	const auto mHPR = static_cast<uint32_t>(name == "HPR");
	const auto mSPR = static_cast<uint32_t>(name == "SPR");
	const auto mGERC = static_cast<uint32_t>(name == "GERC");
	const auto mHER = static_cast<uint32_t>(name == "HER");
	const auto mHER2 = static_cast<uint32_t>(name == "HER2");
	const auto mHPR2 = static_cast<uint32_t>(name == "HPR2");
	const auto mSPR2 = static_cast<uint32_t>(name == "SPR2");

	// Retrieves the kernel from the compiled binary
	tart::kernel_ptr kernel = getGer(device, dtype);
	
	// Sets the kernel arguments
	kernel->setArg(0, static_cast<int>(a_one));
	kernel->setArg(1, static_cast<int>(a_two));
	kernel->setArgAs(dtype, 2, alpha);
	kernel->setArg(3, x_buffer);
	kernel->setArg(4, static_cast<int>(x_offset));
	kernel->setArg(5, static_cast<int>(x_inc));
	kernel->setArg(6, y_buffer);
	kernel->setArg(7, static_cast<int>(y_offset));
	kernel->setArg(8, static_cast<int>(y_inc));
	kernel->setArg(9, a_buffer);
	kernel->setArg(10, static_cast<int>(a_offset));
	kernel->setArg(11, static_cast<int>(a_ld));
	kernel->setArg(12, static_cast<int>(a_is_rowmajor));

	// Launches the kernel
	auto a_one_ceiled = static_cast<uint32_t>(CeilDiv(CeilDiv(a_one, db["WPT"]), db["WGS1"]));
	auto a_two_ceiled = static_cast<uint32_t>(CeilDiv(CeilDiv(a_two, db["WPT"]), db["WGS2"]));
	kernel->enqueue({a_one_ceiled, a_two_ceiled}, {
		static_cast<uint32_t>(db["WGS1"]), static_cast<uint32_t>(db["WGS2"]), static_cast<uint32_t>(db["WPT"]),
		mHPR,
		mSPR,
		mGERC,
		mHER,
		mHER2,
		mHPR2,
		mSPR2
	});
}

// Xger class implementation
Xger::Xger(const tart::DType& dtype, [[maybe_unused]] Queue& queue, const std::string& name) {}

void Xger::DoGer(const Layout layout, const uint32_t m, const uint32_t n, const tart::Scalar& alpha,
                 const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
                 const tart::buffer_ptr& y_buffer, const size_t y_offset, const size_t y_inc,
                 const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld,
                 const std::string& name, [[maybe_unused]] const Databases& db,
                 tart::device_ptr device, const tart::DType& dtype) {
    clblast::DoGer(layout, m, n, alpha, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc,
          a_buffer, a_offset, a_ld, name, db,
          device ? device : device, dtype != tart::dtypes::unknown_type ? dtype : dtype);
}

// =================================================================================================
}	// namespace clblast
