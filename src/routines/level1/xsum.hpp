
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the functional interface for the Xsum routine.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XSUM_H_
#define CLBLAST_ROUTINES_XSUM_H_

#include <cstddef>
#include <string>

#include "database/database.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// Functional implementation of Xsum (sum of values in a vector, non-BLAS)
void DoSum(const tart::DType& dtype, const tart::device_ptr& device,
           const uint32_t n, const tart::buffer_ptr& sum_buffer, const size_t sum_offset,
           const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
           const Databases& db);

// =================================================================================================
}	// namespace clblast

// CLBLAST_ROUTINES_XSUM_H_
#endif
