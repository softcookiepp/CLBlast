
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xasum routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XASUM_H_
#define CLBLAST_ROUTINES_XASUM_H_

#include <cstddef>
#include <string>

#include "routine.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
class Xasum : public Routine
{
	uint32_t mSum;
 public:
	// Constructor
	Xasum(const tart::DType& dtype, Queue& queue, EventPointer event, const std::string& name = "ASUM");

	// Templated-precision implementation of the routine
	void DoAsum(const size_t n, const tart::buffer_ptr& asum_buffer, const size_t asum_offset, const tart::buffer_ptr& x_buffer,
							const size_t x_offset, const size_t x_inc);
};

// =================================================================================================
}	// namespace clblast

// CLBLAST_ROUTINES_XASUM_H_
#endif
