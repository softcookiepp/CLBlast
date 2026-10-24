
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xdotu class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level1/xdotu.hpp"

#include <cstddef>
#include <string>

#include "routines/level1/xdot.hpp"
#include "utilities/backend.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
Xdotu::Xdotu(const tart::DType& dtype, Queue& queue, EventPointer event, const std::string& name)
	: Xdot(dtype, queue, event, name) {}

// =================================================================================================

// The main routine
void Xdotu::DoDotu(const size_t n, const tart::buffer_ptr& dot_buffer, const size_t dot_offset, const tart::buffer_ptr& x_buffer,
											const size_t x_offset, const size_t x_inc, const tart::buffer_ptr& y_buffer, const size_t y_offset,
											const size_t y_inc)
{
	DoDot(n, dot_buffer, dot_offset, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, false);
}

// =================================================================================================
}	// namespace clblast
