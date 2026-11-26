
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the functional interface for the Xcol2im routine.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XCOL2IM_H_
#define CLBLAST_ROUTINES_XCOL2IM_H_

#include <cstddef>
#include <string>

#include "database/database.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// Functional implementation of Xcol2im (column-to-image)
void DoCol2im(const KernelMode kernel_mode, const size_t channels, const uint32_t height, const uint32_t width,
              const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w,
              const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
              const tart::buffer_ptr& col_buffer, const size_t col_offset, const tart::buffer_ptr& im_buffer,
              const size_t im_offset, tart::device_ptr device, const tart::DType& dtype, const Databases& db);

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XCOL2IM_H_
#endif
