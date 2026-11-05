
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xtrsm routine. The implementation is based on ??? (TODO).
// Therefore, this class inherits from the Xgemm class.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XTRSM_H_
#define CLBLAST_ROUTINES_XTRSM_H_

#include <cstddef>
#include <string>

#include "routines/level3/xgemm.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
class Xtrsm : public Xgemm {
 public:
	// Uses methods and variables the Xgemm routine
	using Xgemm::queue_;
	using Xgemm::device_;
	using Xgemm::db_;
	using Xgemm::program_;
	using Xgemm::event_;
	using Xgemm::DoGemm;

	// Constructor
	Xtrsm(const tart::DType& dtype, Queue& queue, EventPointer event, const std::string& name = "TRSM");

	// Templated-precision implementation of the routine
	void DoTrsm(const Layout layout, Side side, Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
							size_t m, size_t n, const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld,
							const tart::buffer_ptr& b_buffer, const size_t b_offset, const size_t b_ld);

	// Implementation of the column-major version
	void TrsmColMajor(const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
										const size_t m, const size_t n, const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer, const size_t a_offset,
										const size_t a_ld, const tart::buffer_ptr& b_buffer, const size_t b_offset, const size_t b_ld);
};

// =================================================================================================
}	// namespace clblast

// CLBLAST_ROUTINES_XTRSM_H_
#endif
