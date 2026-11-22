
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the functional interface for the Xnrm2 routine.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XNRM2_H_
#define CLBLAST_ROUTINES_XNRM2_H_

#include <cstddef>
#include <string>

#include "database/database.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// Functional implementation of Xnrm2 (Euclidean norm)
void DoNrm2(const tart::DType& dtype, const tart::device_ptr& device,
            const size_t n, const tart::buffer_ptr& nrm2_buffer, const size_t nrm2_offset,
            const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
            const Databases& db);

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XNRM2_H_
#endif
