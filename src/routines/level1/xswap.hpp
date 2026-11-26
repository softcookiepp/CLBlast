
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the functional interface for the Xswap routine.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XSWAP_H_
#define CLBLAST_ROUTINES_XSWAP_H_

#include <cstddef>
#include "database/database.hpp"
#include "utilities/backend.hpp"

namespace clblast {

void DoSwap(const tart::DType& dtype, const tart::device_ptr& device,
            const uint32_t n, const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
            const tart::buffer_ptr& y_buffer, const size_t y_offset, const size_t y_inc,
            const Databases& db);

}  // namespace clblast

#endif
