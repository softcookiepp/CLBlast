
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the functional interface for the Xconvgemm routine.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XCONVGEMM_H_
#define CLBLAST_ROUTINES_XCONVGEMM_H_

#include <cstddef>
#include <string>

#include "database/database.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// Functional implementation of Xconvgemm (batched convolution as GEMM)
void DoConvgemm(const KernelMode kernel_mode, const size_t channels, const size_t height, const size_t width,
                const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w,
                const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
                const size_t num_kernels, const size_t batch_count, const tart::buffer_ptr& im_buffer,
                const size_t im_offset, const tart::buffer_ptr& kernel_buffer, const size_t kernel_offset,
                const tart::buffer_ptr& result_buffer, const size_t result_offset, const std::string& name,
                tart::device_ptr device, const tart::DType& dtype, const Databases& db);

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XCONVGEMM_H_
#endif
