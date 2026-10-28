
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xtrsv routine. It uses a block-algorithm and performs small triangular
// forward and backward substitutions on the diagonal parts of the matrix in combination with larger
// GEMV computation on the remainder of the matrix.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XTRSV_H_
#define CLBLAST_ROUTINES_XTRSV_H_

#include <cstddef>
#include <string>

#include "routines/level2/xgemv.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
class Xtrsv : public Xgemv {
 public:
	// Uses the generic matrix-vector routine
	using Xgemv::queue_;
	using Xgemv::device_;
	using Xgemv::db_;
	using Xgemv::program_;
	using Xgemv::event_;
	using Xgemv::DoGemv;

	// Constructor
	Xtrsv(const tart::DType& dtype, Queue& queue, EventPointer event, const std::string& name = "TRSV");

	// Templated-precision implementation of the routine
	void DoTrsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
		const size_t n, const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld,
		const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc);

	// Performs forward or backward substitution on a small triangular matrix
	void Substitution(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
		const size_t n, const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld,
		const tart::buffer_ptr& b_buffer, const size_t b_offset, const size_t b_inc, const tart::buffer_ptr& x_buffer,
		const size_t offset_x, const size_t x_inc);
};

// =================================================================================================
}	// namespace clblast

// CLBLAST_ROUTINES_XTRSV_H_
#endif
