
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the functional interface for the Xim2col routine.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XIM2COL_H_
#define CLBLAST_ROUTINES_XIM2COL_H_

#include <cstddef>
#include <string>

#include "database/database.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// Functional implementation of Xim2col (image-to-column)
void DoIm2col(const KernelMode kernel_mode, const size_t channels, const size_t height, const size_t width,
              const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w,
              const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
              const tart::buffer_ptr& im_buffer, const size_t im_offset, const tart::buffer_ptr& col_buffer,
              const size_t col_offset, tart::device_ptr device, const tart::DType& dtype, const Databases& db);

// =================================================================================================
}	// namespace clblast

// CLBLAST_ROUTINES_XIM2COL_H_
#endif
