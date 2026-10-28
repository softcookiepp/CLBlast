
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xtpmv class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level2/xtpmv.hpp"

#include <cstddef>
#include <string>

#include "routines/level2/xgemv.hpp"
#include "utilities/backend.hpp"
#include "utilities/clblast_exceptions.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
Xtpmv::Xtpmv(const tart::DType& dtype, Queue& queue, EventPointer event, const std::string& name) : Xgemv(dtype, queue, event, name) {}

// =================================================================================================

// The main routine
void Xtpmv::DoTpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose,
											const Diagonal diagonal, const size_t n, const tart::buffer_ptr& ap_buffer, const size_t ap_offset,
											const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc) {
	// Creates a copy of X: a temporary scratch buffer
	const auto x_size = (1 + (n - 1) * x_inc) + x_offset;
	auto scratch_buffer = this->mDevice->allocateBuffer(x_size*this->mDtype.size());
	x_buffer->copyTo(scratch_buffer, 0, 0, x_size*this->mDtype.size());

	// The data is either in the upper or lower triangle
	size_t is_upper = ((triangle == Triangle::kUpper && layout != Layout::kRowMajor) ||
										 (triangle == Triangle::kLower && layout == Layout::kRowMajor));

	// Adds '2' to the parameter if the diagonal is unit
	auto parameter = (diagonal == Diagonal::kUnit) ? is_upper + 2 : is_upper;

	// Runs the generic matrix-vector multiplication, disabling the use of fast vectorized kernels.
	// The specific triangular packed matrix-accesses are implemented in the kernel guarded by the
	// ROUTINE_TPMV define.
	auto fast_kernels = false;
	try {
		MatVec(layout, a_transpose, n, n, tart::Scalar((double)1.0), ap_buffer, ap_offset, n, scratch_buffer, x_offset, x_inc,
			tart::Scalar((double)0.0), x_buffer, x_offset, x_inc, fast_kernels, fast_kernels, parameter, true, 0, 0);
	} catch (BLASError& e) {
		// Returns the proper error code (renames vector Y to X)
		switch (e.status()) {
			case StatusCode::kInvalidVectorY:
				throw BLASError(StatusCode::kInvalidVectorX, e.details());
			case StatusCode::kInvalidIncrementY:
				throw BLASError(StatusCode::kInvalidIncrementX, e.details());
			case StatusCode::kInsufficientMemoryY:
				throw BLASError(StatusCode::kInsufficientMemoryX, e.details());
			default:
				throw;
		}
	}
}

// =================================================================================================
}	// namespace clblast
