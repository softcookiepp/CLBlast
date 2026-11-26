
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the functional Xmax routine (index of maximum, non-BLAS).
//
// =================================================================================================

#include "routines/level1/xmax.hpp"
#include "routines/level1/xamax.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// Functional implementation of Xmax (index of maximum value, non-BLAS)
void DoMax(const tart::DType& dtype, const tart::device_ptr& device,
           const uint32_t n, const tart::buffer_ptr& imax_buffer, const size_t imax_offset,
           const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
           const Databases& db) {
	// max_flag=1, min_flag=0, amin_flag=0 indicates "MAX"
	DoAmax(dtype, device, n, imax_buffer, imax_offset, x_buffer, x_offset, x_inc, 1, 0, 0, db);
}

// =================================================================================================
}  // namespace clblast
