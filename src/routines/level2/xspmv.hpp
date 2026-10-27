
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xspmv routine. It is based on the generalized mat-vec multiplication
// routine (Xgemv). The Xspmv class inherits from the templated class Xgemv, allowing it to call the
// "MatVec" function directly.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XSPMV_H_
#define CLBLAST_ROUTINES_XSPMV_H_

#include <cstddef>
#include <string>

#include "routines/level2/xgemv.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xspmv : public Xgemv {
 public:
	// Uses the generic matrix-vector routine
	using Xgemv::MatVec;

	// Constructor
	Xspmv(Queue& queue, EventPointer event, const std::string& name = "SPMV");

	// Templated-precision implementation of the routine
	void DoSpmv(const Layout layout, const Triangle triangle, const size_t n, const T alpha, const tart::buffer_ptr& ap_buffer,
							const size_t ap_offset, const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
							const T beta, const tart::buffer_ptr& y_buffer, const size_t y_offset, const size_t y_inc);
};

// =================================================================================================
}	// namespace clblast

// CLBLAST_ROUTINES_XSPMV_H_
#endif
