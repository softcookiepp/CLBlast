
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the functional XgemmStridedBatched routine.
//
// =================================================================================================

#include "routines/levelx/xgemmstridedbatched.hpp"

#include <cstddef>
#include <string>
#include <vector>

#include "database/database.hpp"
#include "routine_aot_cache.hpp"
#include "routines/common.hpp"
#include "routines/level3/xgemm.hpp"
#include "utilities/backend.hpp"
#include "utilities/buffer_test.hpp"
#include "utilities/clblast_exceptions.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// borrowing these functions from the test suite just to make it work
static size_t PerBatchSizeA(const Layout layout, const Transpose a_transpose,
    const uint32_t m, const uint32_t n, const uint32_t k, const size_t a_ld)
{
    auto a_rotated = (layout == Layout::kColMajor && a_transpose != Transpose::kNo) ||
                                 (layout == Layout::kRowMajor && a_transpose == Transpose::kNo);
    auto a_two = (a_rotated) ? m : k;
    return a_two * a_ld;
}

static size_t PerBatchSizeB(const Layout layout, const Transpose b_transpose,
    const uint32_t m, const uint32_t n, const uint32_t k, const size_t b_ld)
{
    auto b_rotated = (layout == Layout::kColMajor && b_transpose != Transpose::kNo) ||
                                 (layout == Layout::kRowMajor && b_transpose == Transpose::kNo);
    auto b_two = (b_rotated) ? k : n;
    return b_two * b_ld;
}

static size_t PerBatchSizeC(const Layout layout,
    const uint32_t m, const uint32_t n, const uint32_t k, const size_t c_ld)
{
    auto c_rotated = (layout == Layout::kRowMajor);
    auto c_two = (c_rotated) ? m : n;
    return c_two * c_ld;
}

// The main routine
void DoGemmStridedBatched(const Layout layout, const Transpose a_transpose,
                          const Transpose b_transpose, const uint32_t m, const uint32_t n,
                          const uint32_t k, const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer,
                          const size_t a_offset, const size_t a_ld, const size_t a_stride,
                          const tart::buffer_ptr& b_buffer, const size_t b_offset, const size_t b_ld,
                          const size_t b_stride, const tart::Scalar& beta, const tart::buffer_ptr& c_buffer,
                          const size_t c_offset, const size_t c_ld, const size_t c_stride,
                          const uint32_t batch_count, tart::device_ptr device, const tart::DType& dtype, [[maybe_unused]] const Databases& db)
{
    // Tests for a valid batch count
    if (batch_count < 1) {
        throw BLASError(StatusCode::kInvalidBatchCount);
    }

    // Makes sure the strides are valid
    if (c_stride == 0) {
        throw BLASError(StatusCode::kInvalidDimension);
    }
    
    // right now, using any sort of offset is broken for the Vulkan version of the kernels
    if (a_offset || b_offset || c_offset)
    {
        for (auto batch = size_t{0}; batch < batch_count; ++batch)
        {
            const auto a_batch_offset = a_offset + PerBatchSizeA(layout, a_transpose, m, n, k, a_ld) * batch;
            const auto b_batch_offset = c_offset + PerBatchSizeB(layout, b_transpose, m, n, k, b_ld) * batch;
            const auto c_batch_offset = b_offset + PerBatchSizeC(layout, m, n, k, c_ld) * batch;
            DoGemm(layout, a_transpose, b_transpose,
                m, n, k, alpha, a_buffer, a_batch_offset, a_ld, b_buffer,
                b_batch_offset, b_ld, beta, c_buffer, c_batch_offset, c_ld, tart::buffer_ptr(0), false, db, device, dtype);
        }
        
        return;
    }

    // Two methods to choose from, select which one to run
    const auto do_gemm_direct = clblast::UseDirectKernel(m, n, k, db["XGEMM_MIN_INDIRECT_SIZE"]);
    const auto gemm_kernel_id = (do_gemm_direct) ? 0 : db["GEMMK"];

    // Computes the transpose/conjugate options and sets the a/b/c sizes based on that
    bool a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate;
    size_t a_one, a_two, b_one, b_two, c_one, c_two;
    clblast::ProcessArguments(layout, a_transpose, b_transpose, m, n, k, a_one, a_two, b_one, b_two, c_one, c_two,
                              a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate, gemm_kernel_id);

    // Tests the matrices for validity
    TestStridedBatchedMatrixA(dtype, a_one, a_two, a_buffer, a_offset, a_stride, batch_count, a_ld);
    TestStridedBatchedMatrixB(dtype, b_one, b_two, b_buffer, b_offset, b_stride, batch_count, b_ld);
    TestStridedBatchedMatrixC(dtype, c_one, c_two, c_buffer, c_offset, c_stride, batch_count, c_ld);

    // Selects which version of the batched GEMM to run
    if (do_gemm_direct) {	// single generic kernel
        BatchedGemmDirect(m, n, k, alpha, a_buffer, a_offset, a_ld, a_stride, b_buffer, b_offset, b_ld, b_stride, beta,
                          c_buffer, c_offset, c_ld, c_stride, a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate,
                          b_conjugate, batch_count, device, dtype, db);
    } else {	// pre/post-processing plus a very fast kernel
        BatchedGemmIndirect(m, n, k, alpha, a_buffer, a_offset, a_ld, a_stride, b_buffer, b_offset, b_ld, b_stride, beta,
                            c_buffer, c_offset, c_ld, c_stride, a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate,
                            b_conjugate, a_one, a_two, b_one, b_two, c_one, c_two, batch_count, device, dtype, db);
    }
}

// =================================================================================================

// The indirect version of batched GEMM
void BatchedGemmIndirect(const uint32_t m, const uint32_t n, const uint32_t k, const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer,
                         const size_t a_offset, const size_t a_ld, const size_t a_stride, const tart::buffer_ptr& b_buffer,
                         const size_t b_offset, const size_t b_ld, const size_t b_stride, const tart::Scalar& beta,
                         const tart::buffer_ptr& c_buffer, const size_t c_offset, const size_t c_ld, const size_t c_stride,
                         const bool a_do_transpose, const bool b_do_transpose, const bool c_do_transpose,
                         const bool a_conjugate, const bool b_conjugate, const size_t a_one, const size_t a_two,
                         const size_t b_one, const size_t b_two, const size_t c_one, const size_t c_two, const uint32_t batch_count,
                         tart::device_ptr device, const tart::DType& dtype, [[maybe_unused]] const Databases& db) {
    // Calculates the ceiled versions of m, n, and k
    const auto m_ceiled = Ceil(Ceil(m, db["MWG"]), db["VWM"]);
    const auto n_ceiled = Ceil(Ceil(n, db["NWG"]), db["VWN"]);
    const auto k_ceiled = Ceil(Ceil(k, db["KWG"]), db["VWM"]);

    // Computes the first and second "internal" (ceiled) dimensions
    size_t a_one_i, a_two_i, b_one_i, b_two_i, c_one_i, c_two_i;
    clblast::CalculateInternalDimensions(m, n, k, db["MWG"], db["NWG"], db["KWG"], a_one_i, a_two_i, b_one_i, b_two_i,
                                         c_one_i, c_two_i, db["GEMMK"]);

    // Determines whether or not temporary matrices are needed
    auto a_no_temp = a_one == a_one_i && a_two == a_two_i && a_ld == a_one && !a_do_transpose && !a_conjugate;
    auto b_no_temp = b_one == b_one_i && b_two == b_two_i && b_ld == b_one && !b_do_transpose && !b_conjugate;
    auto c_no_temp = c_one == c_one_i && c_two == c_two_i && c_ld == c_one && !c_do_transpose;

    // Creates the temporary matrices
    const auto a_temp = (a_no_temp) ? a_buffer : device->allocateBuffer(dtype.size() * batch_count * a_one_i * a_two_i);
    const auto b_temp = (b_no_temp) ? b_buffer : device->allocateBuffer(dtype.size() * batch_count * b_one_i * b_two_i);
    const auto c_temp = (c_no_temp) ? c_buffer : device->allocateBuffer(dtype.size() * batch_count * c_one_i * c_two_i);

    // Runs the pre-processing kernel for matrix A
    std::vector<tart::buffer_ptr> barrierBuffers;
    if (!a_no_temp) {
        PadCopyTransposeMatrixStridedBatched(dtype, db, a_one, a_two,
            a_ld, a_offset, a_stride, a_buffer, a_one_i, a_two_i, a_one_i, 0,
            a_one_i * a_two_i, a_temp, true, a_do_transpose, a_conjugate,
            batch_count);
        barrierBuffers.push_back(a_temp);
    }

    // As above, but now for matrix B
    if (!b_no_temp) {
        PadCopyTransposeMatrixStridedBatched(dtype, db, b_one, b_two,
                                             b_ld, b_offset, b_stride, b_buffer, b_one_i, b_two_i, b_one_i, 0,
                                             b_one_i * b_two_i, b_temp, true, b_do_transpose, b_conjugate,
                                             batch_count);
        barrierBuffers.push_back(b_temp);
    }

    // As above, but now for matrix C
    if (!c_no_temp) {
        PadCopyTransposeMatrixStridedBatched(dtype, db, c_one, c_two,
                                             c_ld, c_offset, c_stride, c_buffer, c_one_i, c_two_i, c_one_i, 0,
                                             c_one_i * c_two_i, c_temp, true, c_do_transpose, false, batch_count);
        barrierBuffers.push_back(c_temp);
    }

    // Retrieves the Xgemm kernel from the compiled binary
    tart::kernel_ptr kernel = getGemmStridedBatched(device, dtype, device->getMetadata().subgroupAdd, db["VWM"], db["VWN"]);

    // Sets the kernel arguments
    kernel->setArg(0, static_cast<int>(m_ceiled));
    kernel->setArg(1, static_cast<int>(n_ceiled));
    kernel->setArg(2, static_cast<int>(k_ceiled));
    kernel->setArgAs(dtype, 3, alpha);
    kernel->setArgAs(dtype, 4, beta);
    kernel->setArg(5, a_temp);
    kernel->setArg(6, static_cast<int>(a_one_i));
    kernel->setArg(7, static_cast<int>(a_two_i));
    kernel->setArg(8, b_temp);
    kernel->setArg(9, static_cast<int>(b_one_i));
    kernel->setArg(10, static_cast<int>(b_two_i));
    kernel->setArg(11, c_temp);
    kernel->setArg(12, static_cast<int>(c_one_i));
    kernel->setArg(13, static_cast<int>(c_two_i));

    // Computes the global and local thread sizes
    const auto global = std::vector<size_t>{static_cast<uint32_t>((c_one_i) / db["MWG"]), static_cast<uint32_t>((db["NDIMC"]) / db["NWG"]), static_cast<uint32_t>(batch_count)};
    const auto local = std::vector<uint32_t>{db["MDIMC"], db["NDIMC"], 1};
    
    const bool subgroupSupported = device->getMetadata().subgroupAdd;
    std::vector<uint32_t> spec(subgroupSupported ? 16 : 15);
    spec[0] = db["GEMMK"];
    spec[1] = db["MWG"];
    spec[2] = db["NWG"];
    spec[3] = db["KWG"];
    spec[4] = db["MDIMC"];
    spec[5] = db["NDIMC"];
    spec[6] = db["MDIMA"];
    spec[7] = db["NDIMB"];
    spec[8] = db["KWI"];
    spec[9] = db["STRM"];
    spec[10] = db["STRN"];
    spec[11] = db["SA"];
    spec[12] = db["SB"];
    spec[13] = db["KREG"];
    spec[14] = 1; // there is no db key for this for whatever reason
    if (subgroupSupported)
        spec[15] = 1; // this may actually not need to be a spec constant
    
    // Launches the kernel
    kernel->enqueue(global, spec);

    // Runs the post-processing kernel if needed
    if (!c_no_temp) {
        PadCopyTransposeMatrixStridedBatched(dtype, db, c_one_i, c_two_i, c_one_i, 0,
                                             c_one_i * c_two_i, c_temp, c_one, c_two, c_ld, c_offset, c_stride, c_buffer,
                                             false, c_do_transpose, false, batch_count);
    }
}

// =================================================================================================

// The direct version of batched GEMM
void BatchedGemmDirect(const uint32_t m, const uint32_t n, const uint32_t k, const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer,
                       const size_t a_offset, const size_t a_ld, const size_t a_stride, const tart::buffer_ptr& b_buffer,
                       const size_t b_offset, const size_t b_ld, const size_t b_stride, const tart::Scalar& beta,
                       const tart::buffer_ptr& c_buffer, const size_t c_offset, const size_t c_ld, const size_t c_stride,
                       const bool a_do_transpose, const bool b_do_transpose, const bool c_do_transpose,
                       const bool a_conjugate, const bool b_conjugate, const uint32_t batch_count,
                       tart::device_ptr device, const tart::DType& dtype, [[maybe_unused]] const Databases& db)
{
    // Retrieves the XgemmDirect kernel from the compiled binary
    tart::kernel_ptr kernel = getGemmDirectStridedBatched(device, dtype, db["VWM"], db["VWN"]);

    // Sets the kernel arguments
    kernel->setArg(0, static_cast<int>(m));
    kernel->setArg(1, static_cast<int>(n));
    kernel->setArg(2, static_cast<int>(k));
    kernel->setArgAs(dtype, 3, alpha);
    kernel->setArgAs(dtype, 4, beta);
    kernel->setArg(5, a_buffer);
    kernel->setArg(6, static_cast<int>(a_offset));
    kernel->setArg(7, static_cast<int>(a_ld));
    kernel->setArg(8, static_cast<int>(a_stride));
    kernel->setArg(9, b_buffer);
    kernel->setArg(10, static_cast<int>(b_offset));
    kernel->setArg(11, static_cast<int>(b_ld));
    kernel->setArg(12, static_cast<int>(b_stride));
    kernel->setArg(13, c_buffer);
    kernel->setArg(14, static_cast<int>(c_offset));
    kernel->setArg(15, static_cast<int>(c_ld));
    kernel->setArg(16, static_cast<int>(c_stride));
    kernel->setArg(17, static_cast<int>(c_do_transpose));
    kernel->setArg(18, static_cast<int>(a_conjugate));
    kernel->setArg(19, static_cast<int>(b_conjugate));
    
#if VULKAN_API
    // for workaround to no pointer casting allowed
    kernel->setArg(20, a_buffer);
    kernel->setArg(21, b_buffer);
#endif

    // Computes the global and local thread sizes
    const auto m_ceiled = Ceil(m, db["WGD"]);
    const auto n_ceiled = Ceil(n, db["WGD"]);
    const auto global = std::vector<size_t>{static_cast<uint32_t>(m_ceiled / db["WGD"]), static_cast<uint32_t>(n_ceiled / db["WGD"]), static_cast<uint32_t>(batch_count)};
    
    const std::vector<uint32_t> spec({
        static_cast<uint32_t>(db["WGD"]),
        static_cast<uint32_t>(db["MDIMCD"]),
        static_cast<uint32_t>(db["NDIMCD"]),
        static_cast<uint32_t>(db["MDIMAD"]),
        static_cast<uint32_t>(db["NDIMBD"]),
        static_cast<uint32_t>(db["KWID"]),
        static_cast<uint32_t>(db["PADA"]),
        static_cast<uint32_t>(db["PADB"]),
        a_do_transpose,
        b_do_transpose
    });
    
    // Launches the kernel
    kernel->enqueue(global, spec);
}

// =================================================================================================
}	// namespace clblast
