
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the functional interface for the Xhad routine (Hadamard product).
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XHAD_H_
#define CLBLAST_ROUTINES_XHAD_H_

#include <cstddef>
#include <string>

#include "database/database.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// Functional implementation of Xhad (element-wise vector product)
void DoHad(const size_t n, const tart::Scalar& alpha, const tart::buffer_ptr& x_buffer, const size_t x_offset,
           const size_t x_inc, const tart::buffer_ptr& y_buffer, const size_t y_offset, const size_t y_inc,
           const tart::Scalar& beta, const tart::buffer_ptr& z_buffer, const size_t z_offset, const size_t z_inc,
           tart::device_ptr device, const tart::DType& dtype, const Databases& db);

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XHAD_H_
#endif
