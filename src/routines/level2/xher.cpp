
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the functional Xher routine.
//
// =================================================================================================

#include "routines/level2/xher.hpp"

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
void DoHer(const Layout layout, const Triangle triangle, const uint32_t n, const tart::Scalar& alpha,
           const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
           const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld,
           const bool packed, const std::string& name, [[maybe_unused]] const Databases& db,
           tart::device_ptr device, const tart::DType& dtype) {
	// Makes sure the dimensions are larger than zero
	if (n == 0) {
		throw BLASError(StatusCode::kInvalidDimension);
	}

	// The data is either in the upper or lower triangle
	const auto is_upper = ((triangle == Triangle::kUpper && layout != Layout::kRowMajor) ||
												 (triangle == Triangle::kLower && layout == Layout::kRowMajor));
	const auto is_rowmajor = (layout == Layout::kRowMajor);

	// Tests the matrix and the vectors for validity
	if (packed) {
		TestMatrixAP(dtype, n, a_buffer, a_offset);
	} else {
		TestMatrixA(dtype, n, n, a_buffer, a_offset, a_ld);
	}
	TestVectorX(dtype, n, x_buffer, x_offset, x_inc);
	
	// If alpha is zero an update is not required
	if (alpha.get<double>() == 0.0)
	{
		return;
	}

	// Compute the flags from name
	const auto mHPR = static_cast<uint32_t>(name == "HPR");
	const auto mSPR = static_cast<uint32_t>(name == "SPR");
	const auto mGERC = static_cast<uint32_t>(name == "GERC");
	const auto mHER = static_cast<uint32_t>(name == "HER");
	const auto mHER2 = static_cast<uint32_t>(name == "HER2");
	const auto mHPR2 = static_cast<uint32_t>(name == "HPR2");
	const auto mSPR2 = static_cast<uint32_t>(name == "SPR2");

	// Retrieves the kernel from the compiled binary
	tart::kernel_ptr kernel = getHer(device, dtype);
	
	// Sets the kernel arguments
	kernel->setArg(0, static_cast<int>(n));
	kernel->setArgAs(dtype, 1, alpha);
	kernel->setArg(2, x_buffer);
	kernel->setArg(3, static_cast<int>(x_offset));
	kernel->setArg(4, static_cast<int>(x_inc));
	kernel->setArg(5, a_buffer);
	kernel->setArg(6, static_cast<int>(a_offset));
	kernel->setArg(7, static_cast<int>(a_ld));
	kernel->setArg(8, static_cast<int>(is_upper));
	kernel->setArg(9, static_cast<int>(is_rowmajor));

	// Launches the kernel
	auto global_one = Ceil(CeilDiv(n, db["WPT"]), db["WGS1"]);
	auto global_two = Ceil(CeilDiv(n, db["WPT"]), db["WGS2"]);
	auto global = std::vector<size_t>{static_cast<uint32_t>(global_one / db["WGS1"]), static_cast<uint32_t>(global_two / db["WGS2"])};
	kernel->enqueue(global, {static_cast<uint32_t>(db["WGS1"]), static_cast<uint32_t>(db["WGS2"]), static_cast<uint32_t>(db["WPT"]),
		mHPR,
		mSPR,
		mGERC,
		mHER,
		mHER2,
		mHPR2,
		mSPR2
	});
}

// Xher class implementation
Xher::Xher(const tart::DType& dtype, [[maybe_unused]] Queue& queue, const std::string& name) {}

void Xher::DoHer(const Layout layout, const Triangle triangle, const uint32_t n, const tart::Scalar& alpha,
                 const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
                 const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld,
                 const bool packed, const std::string& name2, const Databases& db,
                 tart::device_ptr device, const tart::DType& dtype) {
    clblast::DoHer(layout, triangle, n, alpha, x_buffer, x_offset, x_inc, a_buffer, a_offset, a_ld,
          packed, name2, db, device ? device : device, dtype != tart::dtypes::unknown_type ? dtype : dtype);
}

// =================================================================================================
}  // namespace clblast
