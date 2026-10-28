
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xhbmv routine. It is based on the generalized mat-vec multiplication
// routine (Xgemv). The Xhbmv class inherits from the templated class Xgemv, allowing it to call the
// "MatVec" function directly.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XHBMV_H_
#define CLBLAST_ROUTINES_XHBMV_H_

#include <cstddef>
#include <string>

#include "routines/level2/xgemv.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
class Xhbmv : public Xgemv {
 public:
	// Uses the generic matrix-vector routine
	using Xgemv::MatVec;

	// Constructor
	Xhbmv(const tart::DType& dtype, Queue& queue, EventPointer event, const std::string& name = "HBMV");

	// Templated-precision implementation of the routine
	void DoHbmv(const Layout layout, const Triangle triangle, const size_t n, const size_t k, const tart::Scalar& alpha,
							const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld, const tart::buffer_ptr& x_buffer,
							const size_t x_offset, const size_t x_inc, const tart::Scalar& beta, const tart::buffer_ptr& y_buffer, const size_t y_offset,
							const size_t y_inc);
};

// =================================================================================================
}	// namespace clblast

// CLBLAST_ROUTINES_XHBMV_H_
#endif
