
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xsymm routine. It is based on the generalized matrix multiplication
// routine (Xgemm). The Xsymm class inherits from the templated class Xgemm, allowing it to call the
// "DoGemm" function directly. The "DoSymm" function first preprocesses the symmetric matrix by
// transforming it into a general matrix, and then calls the regular GEMM code.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XSYMM_H_
#define CLBLAST_ROUTINES_XSYMM_H_

#include <cstddef>
#include <string>

#include "routines/level3/xgemm.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
class Xsymm {
 public:

	// Constructor
	Xsymm(const tart::DType& dtype, Queue& queue, const std::string& name = "SYMM");

	// Templated-precision implementation of the routine
	void DoSymm(const Layout layout, const Side side, const Triangle triangle, const uint32_t m, const uint32_t n,
							const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld,
							const tart::buffer_ptr& b_buffer, const size_t b_offset, const size_t b_ld, const tart::Scalar& beta,
							const tart::buffer_ptr& c_buffer, const size_t c_offset, const size_t c_ld,
							tart::device_ptr device, const tart::DType& dtype, const Databases& db);
};

// =================================================================================================
}	// namespace clblast

// CLBLAST_ROUTINES_XSYMM_H_
#endif
