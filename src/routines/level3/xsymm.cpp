
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xsymm class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level3/xsymm.hpp"

#include <cstddef>
#include <string>
#include <vector>

#include "routines/common.hpp"
#include "routines/level3/xgemm.hpp"
#include "utilities/backend.hpp"
#include "utilities/buffer_test.hpp"
#include "utilities/clblast_exceptions.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
Xsymm::Xsymm(const tart::DType& dtype, [[maybe_unused]] Queue& queue, const std::string& name) {}

// =================================================================================================

// The main routine
void Xsymm::DoSymm(const Layout layout, const Side side, const Triangle triangle, const uint32_t m, const uint32_t n,
											const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld,
											const tart::buffer_ptr& b_buffer, const size_t b_offset, const size_t b_ld, const tart::Scalar& beta,
											const tart::buffer_ptr& c_buffer, const size_t c_offset, const size_t c_ld,
											tart::device_ptr device, const tart::DType& dtype, [[maybe_unused]] const Databases& db) {
	// Makes sure all dimensions are larger than zero
	if ((m == 0) || (n == 0)) {
		throw BLASError(StatusCode::kInvalidDimension);
	}

	// Computes the k dimension. This is based on whether or not the symmetric matrix is A (on the
	// left) or B (on the right) in the Xgemm routine.
	auto k = (side == Side::kLeft) ? m : n;

	// Checks for validity of the squared A matrix
	TestMatrixA(dtype, k, k, a_buffer, a_offset, a_ld);

	// Determines which kernel to run based on the layout (the Xgemm kernel assumes column-major as
	// default) and on whether we are dealing with an upper or lower triangle of the symmetric matrix
	bool is_upper = ((triangle == Triangle::kUpper && layout != Layout::kRowMajor) ||
									 (triangle == Triangle::kLower && layout == Layout::kRowMajor));

	// Temporary buffer for a copy of the symmetric matrix
	auto temp_symm = device->allocateBuffer(dtype.size() * k * k);

	// Creates a general matrix from the symmetric matrix to be able to run the regular Xgemm
	// routine afterwards
	tart::kernel_ptr kernel = (is_upper) ? getConvertSymmetricUpper(device, dtype) : getConvertSymmetricLower(device, dtype);

	// Sets the arguments for the symmetric-to-squared kernel
	kernel->setArg(0, static_cast<int>(k));
	kernel->setArg(1, static_cast<int>(a_ld));
	kernel->setArg(2, static_cast<int>(a_offset));
	kernel->setArg(3, a_buffer);
	kernel->setArg(4, static_cast<int>(k));
	kernel->setArg(5, static_cast<int>(k));
	kernel->setArg(6, static_cast<int>(0));
	kernel->setArg(7, temp_symm);

	// Uses the common padding kernel's thread configuration. This is allowed, since the
	// symmetric-to-squared kernel uses the same parameters.
	auto global = std::vector<uint32_t>{static_cast<uint32_t>(Ceil(CeilDiv(k, db["PAD_WPTX"]), db["PAD_DIMX"])),
																		static_cast<uint32_t>(Ceil(CeilDiv(k, db["PAD_WPTY"]), db["PAD_DIMY"]))};
	auto local = std::vector<uint32_t>{static_cast<uint32_t>(db["PAD_DIMX"]), static_cast<uint32_t>(db["PAD_DIMY"])};
	global[0] = global[0] / local[0];
	global[1] = global[1] / local[1];
	std::vector<uint32_t> spec = {
		static_cast<uint32_t>(db["COPY_DIMX"]),
		static_cast<uint32_t>(db["COPY_DIMY"]),
		static_cast<uint32_t>(db["COPY_WPT"]),
		static_cast<uint32_t>(db["PAD_DIMX"]),
		static_cast<uint32_t>(db["PAD_DIMY"]),
		static_cast<uint32_t>(db["PAD_WPTX"]),
		static_cast<uint32_t>(db["PAD_WPTY"]),
		static_cast<uint32_t>(db["TRA_DIM"]),
		static_cast<uint32_t>(db["TRA_PAD"]),
		static_cast<uint32_t>(db["TRA_SHUFFLE"]),
		static_cast<uint32_t>(db["PADTRA_TILE"]),
		static_cast<uint32_t>(db["PADTRA_WPT"]),
		static_cast<uint32_t>(db["PADTRA_PAD"]),
		0, 0, 0, 0
	};
	kernel->enqueue(global, spec);

	// Runs the regular Xgemm code with either "C := AB+C" or ...
	if (side == Side::kLeft) {
		DoGemm(layout, Transpose::kNo, Transpose::kNo, m, n, k, alpha, temp_symm, 0, k, b_buffer, b_offset, b_ld, beta,
					 c_buffer, c_offset, c_ld, tart::buffer_ptr{}, false, db, device, dtype);
	}

	// ... with "C := BA+C". Note that A and B are now reversed.
	else {
		try {
			DoGemm(layout, Transpose::kNo, Transpose::kNo, m, n, k, alpha, b_buffer, b_offset, b_ld, temp_symm, 0, k, beta,
						 c_buffer, c_offset, c_ld, tart::buffer_ptr{}, false, db, device, dtype);
		} catch (BLASError& e) {
			// A and B are now reversed, so also reverse the error codes returned from the Xgemm routine
			switch (e.status()) {
				case StatusCode::kInvalidMatrixA:
					throw BLASError(StatusCode::kInvalidMatrixB, e.details());
				case StatusCode::kInvalidMatrixB:
					throw BLASError(StatusCode::kInvalidMatrixA, e.details());
				case StatusCode::kInvalidLeadDimA:
					throw BLASError(StatusCode::kInvalidLeadDimB, e.details());
				case StatusCode::kInvalidLeadDimB:
					throw BLASError(StatusCode::kInvalidLeadDimA, e.details());
				case StatusCode::kInsufficientMemoryA:
					throw BLASError(StatusCode::kInsufficientMemoryB, e.details());
				case StatusCode::kInsufficientMemoryB:
					throw BLASError(StatusCode::kInsufficientMemoryA, e.details());
				default:
					throw;
			}
		}
	}
}

// =================================================================================================
}	// namespace clblast
