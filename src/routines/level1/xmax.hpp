
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the functional interface for the Xmax routine.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XMAX_H_
#define CLBLAST_ROUTINES_XMAX_H_

#include <cstddef>
#include <string>

#include "database/database.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// Functional implementation of Xmax (index of maximum value, non-BLAS)
void DoMax(const tart::DType& dtype, const tart::device_ptr& device,
           const size_t n, const tart::buffer_ptr& imax_buffer, const size_t imax_offset,
           const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
           const Databases& db);

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XMAX_H_
#endif
