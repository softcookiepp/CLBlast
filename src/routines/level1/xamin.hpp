
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the functional interface for the Xamin routine.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XAMIN_H_
#define CLBLAST_ROUTINES_XAMIN_H_

#include <cstddef>
#include <string>

#include "database/database.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// Functional implementation of Xamin (index of absolute minimum value, non-BLAS)
void DoAmin(const tart::DType& dtype, const tart::device_ptr& device,
            const uint32_t n, const tart::buffer_ptr& imin_buffer, const size_t imin_offset,
            const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
            const Databases& db);

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XAMIN_H_
#endif
