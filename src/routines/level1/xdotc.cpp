
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the functional Xdotc routine.
//
// =================================================================================================

#include "routines/level1/xdotc.hpp"
#include "routines/level1/xdot.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// Functional implementation of Xdotc (dot product with conjugation)
void DoDotc(const tart::DType& dtype, const tart::device_ptr& device,
            const uint32_t n, const tart::buffer_ptr& dot_buffer, const size_t dot_offset,
            const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
            const tart::buffer_ptr& y_buffer, const size_t y_offset, const size_t y_inc,
            const Databases& db) {
	DoDot(dtype, device, n, dot_buffer, dot_offset, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, true, db);
}
// =================================================================================================
}	// namespace clblast
