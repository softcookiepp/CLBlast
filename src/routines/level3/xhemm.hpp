
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xhemm routine. It is based on the generalized matrix multiplication
// routine (Xgemm). The implementation is very similar to the Xsymm routine.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XHEMM_H_
#define CLBLAST_ROUTINES_XHEMM_H_

#include <cstddef>
#include <string>

#include "routines/level3/xgemm.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
class Xhemm {
 public:

	// Constructor
	Xhemm(const tart::DType& dtype, Queue& queue, EventPointer event, const std::string& name = "HEMM");

	// Templated-precision implementation of the routine
	void DoHemm(const Layout layout, const Side side, const Triangle triangle, const size_t m, const size_t n,
							const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld,
							const tart::buffer_ptr& b_buffer, const size_t b_offset, const size_t b_ld, const tart::Scalar& beta,
							const tart::buffer_ptr& c_buffer, const size_t c_offset, const size_t c_ld,
							tart::device_ptr device, const tart::DType& dtype, const Databases& db);
};

// =================================================================================================
}	// namespace clblast

// CLBLAST_ROUTINES_XHEMM_H_
#endif
