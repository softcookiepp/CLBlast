
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the functional Xsum routine (sum of values, non-BLAS).
//
// =================================================================================================

#include "routines/level1/xsum.hpp"
#include "routines/level1/xasum.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// Functional implementation of Xsum (sum of values in a vector, non-BLAS)
void DoSum(const tart::DType& dtype, const tart::device_ptr& device,
           const uint32_t n, const tart::buffer_ptr& sum_buffer, const size_t sum_offset,
           const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
           const Databases& db)
{
	// sum_flag=1 indicates "SUM" instead of "ASUM"
	DoAsum(dtype, device, n, sum_buffer, sum_offset, x_buffer, x_offset, x_inc, 1, db);
}

// =================================================================================================
}	// namespace clblast
