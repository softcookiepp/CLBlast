
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the functional Xinvert routine.
//
// =================================================================================================

#include "routines/levelx/xinvert.hpp"

#include <assert.h>

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

// Inverts diagonal square blocks of a matrix
void InvertMatrixDiagonalBlocks(const Layout layout, const Triangle triangle, const Diagonal diag,
                                const uint32_t n, const uint32_t block_size, const tart::buffer_ptr& src,
                                const uint32_t offset, const size_t ld_src, tart::buffer_ptr& dest,
                                const tart::device_ptr& device, const tart::DType& dtype, const Databases& db)
{
    // Makes sure all dimensions are larger than zero
    if ((block_size == 0) || (n == 0)) {
        throw BLASError(StatusCode::kInvalidDimension);
    }

    // Some parts of this kernel are not tunable and thus require some minimal Vulkan properties
    if (device->getMetadata().maxSubgroupSize < 16) {	// minimum of total local work size of 16
        throw RuntimeErrorCode(StatusCode::kNotImplemented);
    }

    // Helper variables
    const auto internal_block_size = static_cast<size_t>(db["INTERNAL_BLOCK_SIZE"]);
    if (internal_block_size != 16) {
        throw RuntimeErrorCode(StatusCode::kNotImplemented);	// e.g. Apple CPU OpenCL with a WGS of 1
    }	// when barriers are present
    const auto num_blocks = CeilDiv(n, block_size);
    const auto num_internal_blocks = CeilDiv(n, internal_block_size);
    const auto unit_diagonal = (diag == Diagonal::kUnit) ? true : false;

    // This routine only supports block sizes which are a multiple of the internal block size and
    // block sizes up to and including 128
    if ((block_size % internal_block_size != 0) || (block_size > 128)) {
        throw BLASError(StatusCode::kUnknownError);
    }

    // Checks for validity of the source and destination matrices
    TestMatrixA(dtype, n, n, src, offset, ld_src);
    TestMatrixB(dtype, block_size, num_blocks * block_size, dest, 0, block_size);

    // Determines which kernels to run based on the layout
    const bool is_upper = ((triangle == Triangle::kUpper && layout != Layout::kRowMajor) ||
                           (triangle == Triangle::kLower && layout == Layout::kRowMajor));

    // Fills the output buffer with zeros
    FillMatrix(dtype, block_size,
               num_blocks * block_size, block_size, 0, dest, tart::Scalar::zero(), 16);

    // Inverts the diagonal IB by IB inner blocks of the matrix: one block per work-group
    tart::kernel_ptr kernel = getInvertDiagonalBlock(device, dtype);
    kernel->setArg(0, static_cast<int>(n));
    kernel->setArg(1, src);
    kernel->setArg(2, static_cast<int>(offset));
    kernel->setArg(3, static_cast<int>(ld_src));
    kernel->setArg(4, dest);
    kernel->setArg(5, static_cast<int>(block_size));
    kernel->setArg(6, static_cast<int>(unit_diagonal));
    kernel->setArg(7, static_cast<int>(is_upper));
    const auto local_invert = std::vector<uint32_t>{internal_block_size};
    auto global_invert = std::vector<size_t>{static_cast<uint32_t>(num_internal_blocks * internal_block_size)};
    global_invert[0] = global_invert[0] / local_invert[0];
    std::vector<uint32_t> spec = {
        static_cast<uint32_t>(db["INTERNAL_BLOCK_SIZE"]),
        static_cast<uint32_t>(db["LOCALPAD"]),
        static_cast<uint32_t>(db["TMMWGSX"]),
        static_cast<uint32_t>(db["TMMWGSY"]),
        0,
        static_cast<uint32_t>(is_upper)
    };
    kernel->enqueue(global_invert, spec);
    
    // Builds up block_size x block_size blocks
    for (auto current_size = internal_block_size; current_size < block_size; current_size *= 2)
    {
        assert(current_size == 16 || current_size == 32 || current_size == 64);
        
        std::vector<uint32_t> spec2 = {
            static_cast<uint32_t>(db["INTERNAL_BLOCK_SIZE"]),
            static_cast<uint32_t>(db["LOCALPAD"]),
            static_cast<uint32_t>(db["TMMWGSX"]),
            static_cast<uint32_t>(db["TMMWGSY"]),
            static_cast<uint32_t>(current_size),
            static_cast<uint32_t>(is_upper)
        };

        // Emulates a 3D grid: NX * (NY * npages)
        const auto npages = CeilDiv(n, current_size * 2);
        const auto local0 = (current_size <= 32) ? current_size / 4 : 16;
        const auto local = std::vector<uint32_t>{local0, 4};
        auto global = std::vector<size_t>{static_cast<uint32_t>(Ceil(current_size / local[1], local[0])), static_cast<uint32_t>(Ceil(npages * (current_size / 16) * local[1], local[1]))};
        global[0] = global[0] / local0;
        global[1] = global[1] / 4;

        // Part 1
        tart::kernel_ptr kernel1 = getInvertTripleMatmul(device, dtype);
        kernel1->setArg(0, static_cast<int>(n));
        kernel1->setArg(1, src);
        kernel1->setArg(2, static_cast<int>(offset));
        kernel1->setArg(3, static_cast<int>(ld_src));
        kernel1->setArg(4, dest);
        kernel1->setArg(5, static_cast<int>(current_size));
        kernel1->setArg(6, static_cast<int>(npages));
        kernel1->setArg(7, static_cast<int>(block_size));
        kernel1->enqueue(global, spec2);

        // Part 2
        tart::kernel_ptr kernel2 = getInvertTripleMatmulPart2(device, dtype);
        kernel2->setArg(0, static_cast<int>(n));
        kernel2->setArg(1, dest);
        kernel2->setArg(2, static_cast<int>(current_size));
        kernel2->setArg(3, static_cast<int>(npages));
        kernel2->setArg(4, static_cast<int>(block_size));
        kernel2->enqueue(global, spec2);
        
        // Exit in case we reach beyond the bounds of the input matrix
        if (current_size * 2 >= n) {
            break;
        }
    }
}

// =================================================================================================
}  // namespace clblast
