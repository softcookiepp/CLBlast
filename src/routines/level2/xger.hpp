
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xger routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XGER_H_
#define CLBLAST_ROUTINES_XGER_H_

#include <cstddef>
#include <string>

#include "routine.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
class Xger : public Routine
{
	uint32_t mHPR;
	uint32_t mSPR;
	uint32_t mGERC;
	uint32_t mHER;
	uint32_t mHER2;
	uint32_t mHPR2;
	uint32_t mSPR2;
 public:
	// Constructor
	Xger(const tart::DType& dtype, Queue& queue, EventPointer event, const std::string& name = "GER");

	// Templated-precision implementation of the routine
	void DoGer(const Layout layout, const size_t m, const size_t n, const tart::Scalar& alpha, const tart::buffer_ptr& x_buffer,
						 const size_t x_offset, const size_t x_inc, const tart::buffer_ptr& y_buffer, const size_t y_offset,
						 const size_t y_inc, const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld);
};

// =================================================================================================
}	// namespace clblast

// CLBLAST_ROUTINES_XGER_H_
#endif
