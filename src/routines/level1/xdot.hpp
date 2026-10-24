
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xdot routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XDOT_H_
#define CLBLAST_ROUTINES_XDOT_H_

#include <cstddef>
#include <string>

#include "routine.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
#if 1
	class Xdot : public Routine
	{
	 public:
		// Constructor
		Xdot(const tart::DType& dtype, Queue& queue, EventPointer event, const std::string& name = "DOT");

		// Templated-precision implementation of the routine
		void DoDot(const size_t n, const tart::buffer_ptr& dot_buffer, const size_t dot_offset, const tart::buffer_ptr& x_buffer,
				 const size_t x_offset, const size_t x_inc, const tart::buffer_ptr& y_buffer, const size_t y_offset,
				 const size_t y_inc, const bool do_conjugate = false);
	};
#else
	template <typename T>
	class Xdot : public Routine
	{
	 public:
		// Constructor
		Xdot(Queue& queue, EventPointer event, const std::string& name = "DOT");

		// Templated-precision implementation of the routine
		void DoDot(const size_t n, const Buffer<T>& dot_buffer, const size_t dot_offset, const Buffer<T>& x_buffer,
				 const size_t x_offset, const size_t x_inc, const Buffer<T>& y_buffer, const size_t y_offset,
				 const size_t y_inc, const bool do_conjugate = false);
	};
#endif

// =================================================================================================
}	// namespace clblast

// CLBLAST_ROUTINES_XDOT_H_
#endif
