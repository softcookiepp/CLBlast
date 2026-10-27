
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xsum routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XSUM_H_
#define CLBLAST_ROUTINES_XSUM_H_

#include <cstddef>
#include <string>

#include "routines/level1/xasum.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
class Xsum : public Xasum {
 public:
	// Members and methods from the base class
	using Xasum::DoAsum;

	// Constructor
	Xsum(const tart::DType& dtype, Queue& queue, EventPointer event, const std::string& name = "SUM") : Xasum(dtype, queue, event, name) {}

	// Forwards to the regular absolute version. The implementation difference is realised in the
	// kernel through a pre-processor macro based on the name of the routine.
	void DoSum(const size_t n, const tart::buffer_ptr& sum_buffer, const size_t sum_offset, const tart::buffer_ptr& x_buffer,
						 const size_t x_offset, const size_t x_inc)
	{
		DoAsum(n, sum_buffer, sum_offset, x_buffer, x_offset, x_inc);
	}
};

// =================================================================================================
}	// namespace clblast

// CLBLAST_ROUTINES_XSUM_H_
#endif
