
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the functional interface for the Xcopy routine.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XCOPY_H_
#define CLBLAST_ROUTINES_XCOPY_H_

#include <cstddef>
#include "database/database.hpp"
#include "utilities/backend.hpp"

namespace clblast {

// Functional implementation of Xcopy (y = x)
void DoCopy(const tart::DType& dtype, const tart::device_ptr& device,
            const size_t n, const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
            const tart::buffer_ptr& y_buffer, const size_t y_offset, const size_t y_inc,
            const Databases& db);

}  // namespace clblast

#endif
