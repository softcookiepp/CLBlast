
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xtpmv routine. It is based on the generalized mat-vec multiplication
// routine (Xgemv). The Xtpmv class inherits from the templated class Xgemv, allowing it to call the
// "MatVec" function directly.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XTPMV_H_
#define CLBLAST_ROUTINES_XTPMV_H_

#include <cstddef>
#include <string>

#include "routines/level2/xgemv.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
class Xtpmv : public Xgemv {
 public:
	// Uses the generic matrix-vector routine
	using Xgemv::queue_;
	using Xgemv::MatVec;

	// Constructor
	Xtpmv(const tart::DType& dtype, Queue& queue, EventPointer event, const std::string& name = "TPMV");

	// Templated-precision implementation of the routine
	void DoTpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
							const size_t n, const tart::buffer_ptr& ap_buffer, const size_t ap_offset, const tart::buffer_ptr& x_buffer,
							const size_t x_offset, const size_t x_inc);
};

// =================================================================================================
}	// namespace clblast

// CLBLAST_ROUTINES_XTPMV_H_
#endif
