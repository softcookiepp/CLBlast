
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the functional XaxpyBatched routine.
//
// =================================================================================================

#include "routines/levelx/xaxpybatched.hpp"

#include <cstddef>
#include <string>
#include <vector>

#include "database/database.hpp"
#include "routine_aot_cache.hpp"
#include "routines/common.hpp"
#include "utilities/backend.hpp"
#include "utilities/buffer_test.hpp"
#include "utilities/clblast_exceptions.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// The main routine
void DoAxpyBatched(const uint32_t n, const tart::buffer_ptr& alphas_device, const tart::buffer_ptr& x_buffer,
                   const std::vector<uint32_t>& x_offsets, const size_t x_inc, const tart::buffer_ptr& y_buffer,
                   const std::vector<uint32_t>& y_offsets, const size_t y_inc,
                   const uint32_t batch_count, tart::device_ptr device, const tart::DType& dtype, const Databases& db)
{
    // Makes sure all dimensions are larger than zero
    if (n == 0) {
        throw BLASError(StatusCode::kInvalidDimension);
    }

    // Tests the vectors for validity
    for (auto batch = size_t{0}; batch < batch_count; ++batch) {
        TestVectorX(dtype, n, x_buffer, x_offsets[batch], x_inc);
        TestVectorY(dtype, n, y_buffer, y_offsets[batch], y_inc);
    }

    // Upload the arguments to the device
    auto x_offsets_int = std::vector<int>(batch_count);
    auto y_offsets_int = std::vector<int>(batch_count);
    for (auto batch = size_t{0}; batch < batch_count; ++batch)
    {
        x_offsets_int[batch] = static_cast<int>(x_offsets[batch]);
        y_offsets_int[batch] = static_cast<int>(y_offsets[batch]);
    }
    auto x_offsets_device = device->allocateBuffer(x_offsets_int.size()*sizeof(int));
    x_offsets_device->copyIn(x_offsets_int);
    auto y_offsets_device = device->allocateBuffer(y_offsets_int.size()*sizeof(int));
    y_offsets_device->copyIn(y_offsets_int);

    // Retrieves the Xaxpy kernel from the compiled binary
    tart::kernel_ptr kernel = getAxpyBatched(device, dtype);

    // Sets the kernel arguments
    kernel->setArg(0, static_cast<int>(n));
    kernel->setArg(1, alphas_device);
    kernel->setArg(2, x_buffer);
    kernel->setArg(3, x_offsets_device);
    kernel->setArg(4, static_cast<int>(x_inc));
    kernel->setArg(5, y_buffer);
    kernel->setArg(6, y_offsets_device);
    kernel->setArg(7, static_cast<int>(y_inc));

    // Launches the kernel
    auto n_ceiled = Ceil(n, db["WGS"] * db["WPT"]);
    auto global = std::vector<uint32_t>{n_ceiled / db["WPT"], batch_count};
    auto local = std::vector<uint32_t>{db["WGS"], 1};
    kernel->enqueue({static_cast<uint32_t>(global[0]), static_cast<uint32_t>(global[1]), 1}, {static_cast<uint32_t>(db["WGS"]), static_cast<uint32_t>(db["WPT"])});
}

// =================================================================================================
}	// namespace clblast
