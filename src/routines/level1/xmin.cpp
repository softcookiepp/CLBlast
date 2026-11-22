
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the functional Xmin routine (index of minimum, non-BLAS).
//
// =================================================================================================

#include "routines/level1/xmin.hpp"
#include "routines/level1/xamax.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// Functional implementation of Xmin (index of minimum value, non-BLAS)
void DoMin(const tart::DType& dtype, const tart::device_ptr& device,
           const size_t n, const tart::buffer_ptr& imin_buffer, const size_t imin_offset,
           const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
           const Databases& db) {
	// max_flag=0, min_flag=1, amin_flag=0 indicates "MIN"
	DoAmax(dtype, device, n, imin_buffer, imin_offset, x_buffer, x_offset, x_inc, 0, 1, 0, db);
}

// =================================================================================================
}  // namespace clblast
