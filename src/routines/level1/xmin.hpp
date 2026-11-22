
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the functional interface for the Xmin routine.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XMIN_H_
#define CLBLAST_ROUTINES_XMIN_H_

#include <cstddef>
#include <string>

#include "database/database.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// Functional implementation of Xmin (index of minimum value, non-BLAS)
void DoMin(const tart::DType& dtype, const tart::device_ptr& device,
           const size_t n, const tart::buffer_ptr& imin_buffer, const size_t imin_offset,
           const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
           const Databases& db);

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XMIN_H_
#endif
