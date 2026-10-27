
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xamax routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XAMAX_H_
#define CLBLAST_ROUTINES_XAMAX_H_

#include <cstddef>
#include <string>

#include "routine.hpp"
#include "utilities/backend.hpp"

namespace clblast
{
// =================================================================================================

// See comment at top of file for a description of the class
class Xamax : public Routine
{
	
	uint32_t mMax;
	uint32_t mMin;
	uint32_t mAmin;
 public:
	// Constructor
	Xamax(const tart::DType& dtype, Queue& queue, EventPointer event, const std::string& name = "AMAX");

	// Templated-precision implementation of the routine
	void DoAmax(const size_t n, const tart::buffer_ptr& imax_buffer, const size_t imax_offset,
							const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc);
};

// =================================================================================================
}	// namespace clblast

// CLBLAST_ROUTINES_XAMAX_H_
#endif
