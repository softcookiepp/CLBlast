
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the functional Xconvgemm routine.
//
// =================================================================================================

#include "routines/levelx/xconvgemm.hpp"

#include <assert.h>

#include <cstddef>
#include <string>
#include <vector>

#include "database/database.hpp"
#include "routine_aot_cache.hpp"
#include "routines/common.hpp"
#include "routines/levelx/xim2col.hpp"
#include "utilities/backend.hpp"
#include "utilities/buffer_test.hpp"
#include "utilities/clblast_exceptions.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

void DoConvgemm(const KernelMode kernel_mode, const size_t channels, const size_t height,
                const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h,
                const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h,
                const size_t dilation_w, const size_t num_kernels, const size_t batch_count,
                const tart::buffer_ptr& im_buffer, const size_t im_offset, const tart::buffer_ptr& kernel_buffer,
                const size_t kernel_offset, const tart::buffer_ptr& result_buffer, const size_t result_offset,
                const std::string& name, tart::device_ptr device, const tart::DType& dtype, const Databases& db)
{
    // Tests for a valid batch count
    if (batch_count == 0) {
        throw BLASError(StatusCode::kInvalidBatchCount);
    }

    // Makes sure all dimensions are larger than zero
    if ((channels == 0) || (height == 0) || (width == 0) || (num_kernels == 0)) {
        throw BLASError(StatusCode::kInvalidDimension);
    }

    // Sets the output height and width
    const auto size_h = height + 2 * pad_h;
    const auto padding_h = dilation_h * (kernel_h - 1) + 1;
    const auto output_h = (size_h >= padding_h) ? (size_h - padding_h) / stride_h + 1 : 1;
    const auto size_w = width + 2 * pad_w;
    const auto padding_w = dilation_w * (kernel_w - 1) + 1;
    const auto output_w = (size_w >= padding_w) ? (size_w - padding_w) / stride_w + 1 : 1;

    // Sets other useful variables
    const auto patch_size = kernel_h * kernel_w * channels;
    const auto num_patches = output_h * output_w;
    
    // Possible approach: im2col + GEMM
    //         result = GEMM(im2col(image), kernel)
    tart::buffer_ptr col_buffer = nullptr;
    
    // If this is called with kWithIm2Col, we need to do im2col for each batch
    if (name == "CONVGEMM_WITH_IM2COL")
    {
        // Temporary col matrix
        const auto col_size = patch_size * num_patches * batch_count;
        col_buffer = device->allocateBuffer(dtype.size() * col_size);

        // Loops over each batch
        for (auto batch_id = size_t{0}; batch_id < batch_count; ++batch_id) {
            // im2col
            const auto im_batch_offset = batch_id * channels * height * width + im_offset;
            const auto col_batch_offset = batch_id * patch_size * num_patches;
            // Get db for im2col (uses same kernel set as Copy)
            const auto im2col_db = GetDatabases(device, {"Copy"}, dtype);
            DoIm2col(kernel_mode, channels, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
                     dilation_h, dilation_w, im_buffer, im_batch_offset, col_buffer, col_batch_offset, device, dtype, im2col_db);
        }
    }

    // Strided batched GEMM: C (result) = alpha (1) * A (col) * B (kernel) + beta (0) * C (result)
    const auto col_stride = patch_size * num_patches;
    const auto result_stride = num_kernels * output_h * output_w;

    // Tests the matrices for validity
    TestMatrixB(dtype, patch_size, num_kernels, kernel_buffer, kernel_offset, patch_size);
    for (auto batch = size_t{0}; batch < batch_count; ++batch) {
        if (name == "CONVGEMM_WITH_IM2COL") {
            TestMatrixA(dtype, num_patches, patch_size, col_buffer, col_stride * batch, num_patches);
        } else {
            // TODO: check for valid image tensor
        }
        TestMatrixC(dtype, num_patches, num_kernels, result_buffer, result_offset + result_stride * batch, num_patches);
    }
    
    // Retrieves the proper XgemmDirect kernel from the compiled binary
    const bool with_im2col = (name == "CONVGEMM_WITH_IM2COL");
    tart::kernel_ptr kernel = with_im2col ?
        getConvgemmWithIm2col(device, dtype, db["VWMD"], db["VWND"]):
        getConvgemm(device, dtype, db["VWMD"], db["VWND"]);
    
    // Sets the kernel arguments
    kernel->setArg(0, static_cast<int>(num_patches));
    kernel->setArg(1, static_cast<int>(num_kernels));
    kernel->setArg(2, static_cast<int>(patch_size));
    kernel->setArg(3, kernel_buffer);
    kernel->setArg(4, static_cast<int>(kernel_offset));
    kernel->setArg(5, result_buffer);
    kernel->setArg(6, static_cast<int>(result_offset));
    kernel->setArg(7, static_cast<int>(result_stride));
    if (with_im2col) {
        // both buffers need scalar casting
        kernel->setArg(23, kernel_buffer);
        kernel->setArg(24, col_buffer);
        kernel->setArg(9, static_cast<int>(0));
        kernel->setArg(10, static_cast<int>(col_stride));
    }
    if (name == "CONVGEMM_SINGLE_KERNEL") {
        kernel->setArg(23, kernel_buffer);
        kernel->setArg(8, im_buffer);
        kernel->setArg(9, static_cast<int>(im_offset));
        kernel->setArg(10, static_cast<int>(height));
        kernel->setArg(11, static_cast<int>(width));
        kernel->setArg(12, static_cast<int>(channels));
        kernel->setArg(13, static_cast<int>(kernel_h));
        kernel->setArg(14, static_cast<int>(kernel_w));
        kernel->setArg(15, static_cast<int>(pad_h));
        kernel->setArg(16, static_cast<int>(pad_w));
        kernel->setArg(17, static_cast<int>(stride_h));
        kernel->setArg(18, static_cast<int>(stride_w));
        kernel->setArg(19, static_cast<int>(dilation_h));
        kernel->setArg(20, static_cast<int>(dilation_w));
        kernel->setArg(21, static_cast<int>(output_h));
        kernel->setArg(22, static_cast<int>(output_w));
    }

    // Computes the global and local thread sizes
    const auto m_ceiled = Ceil(num_patches, db["WGD"]);
    const auto n_ceiled = Ceil(num_kernels, db["WGD"]);
    const auto global = std::vector<uint32_t>{m_ceiled / db["WGD"], (n_ceiled) / db["WGD"], batch_count};
    
    std::vector<uint32_t> spec;
    if (with_im2col)
    {
        spec = {
            db["WGD"],
            db["MDIMCD"],
            db["NDIMCD"],
            
            db["MDIMAD"],
            db["NDIMBD"],
            db["KWID"],
            db["PADA"],
            db["PADB"]
        };
    }
    else
    {
        spec = {
            db["WGD"],
            db["MDIMCD"],
            db["NDIMCD"],
            
            db["MDIMAD"],
            db["NDIMBD"],
            db["KWID"],
            db["PADA"],
            db["PADB"],
            0, 0,
            static_cast<uint32_t>(kernel_mode == KernelMode::kConvolution)
        };
    }

    // Launches the kernel
    kernel->enqueue(global, spec);
}

// =================================================================================================
}  // namespace clblast
