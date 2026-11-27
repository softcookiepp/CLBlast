
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the functional XgemmBatched routine.
//
// =================================================================================================

#include "routines/levelx/xgemmbatched.hpp"

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

// The main routine
void DoGemmBatched(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                   const uint32_t m, const uint32_t n, const uint32_t k, const tart::buffer_ptr& alphas_device,
                   const tart::buffer_ptr& a_buffer, const std::vector<uint32_t>& a_offsets, const size_t a_ld,
                   const tart::buffer_ptr& b_buffer, const std::vector<uint32_t>& b_offsets, const size_t b_ld,
                   const tart::buffer_ptr& betas_device, const tart::buffer_ptr& c_buffer,
                   const std::vector<uint32_t>& c_offsets, const size_t c_ld, const uint32_t batch_count,
                   tart::device_ptr device, const tart::DType& dtype, const Databases& db)
{
    // Tests for a valid batch count
    if ((batch_count < 1) || (alphas_device->getSize()/dtype.size() != batch_count) || (betas_device->getSize()/dtype.size() != batch_count) ||
            (a_offsets.size() != batch_count) || (b_offsets.size() != batch_count) || (c_offsets.size() != batch_count)) {
        throw BLASError(StatusCode::kInvalidBatchCount);
    }

    // Two methods to choose from, select which one to run
#if 1
    const bool do_gemm_direct = true;
#else
    const auto do_gemm_direct = clblast::UseDirectKernel(m, n, k, db["XGEMM_MIN_INDIRECT_SIZE"]);
#endif
    const auto gemm_kernel_id = (do_gemm_direct) ? 0 : db["GEMMK"];
    
    // Computes the transpose/conjugate options and sets the a/b/c sizes based on that
    bool a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate;
    size_t a_one, a_two, b_one, b_two, c_one, c_two;
    clblast::ProcessArguments(layout, a_transpose, b_transpose, m, n, k, a_one, a_two, b_one, b_two, c_one, c_two,
                              a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate, gemm_kernel_id);

    // Tests the matrices for validity
    TestBatchedMatrixA(dtype, a_one, a_two, a_buffer, a_offsets, a_ld, false);
    TestBatchedMatrixB(dtype, b_one, b_two, b_buffer, b_offsets, b_ld, false);
    TestBatchedMatrixC(dtype, c_one, c_two, c_buffer, c_offsets, c_ld);

    // Converts the offset to integers
    auto a_offsets_int = std::vector<int>(batch_count);
    auto b_offsets_int = std::vector<int>(batch_count);
    auto c_offsets_int = std::vector<int>(batch_count);
    for (auto batch = size_t{0}; batch < batch_count; ++batch) {
        a_offsets_int[batch] = static_cast<int>(a_offsets[batch]);
        b_offsets_int[batch] = static_cast<int>(b_offsets[batch]);
        c_offsets_int[batch] = static_cast<int>(c_offsets[batch]);
    }

    // Selects which version of the batched GEMM to run
    if (do_gemm_direct) {	// single generic kernel
        BatchedGemmDirect(m, n, k, alphas_device, a_buffer, a_offsets_int, a_ld, b_buffer, b_offsets_int, b_ld,
                          betas_device, c_buffer, c_offsets_int, c_ld, a_do_transpose, b_do_transpose, c_do_transpose,
                          a_conjugate, b_conjugate, batch_count, device, dtype, db);
    } else {	// pre/post-processing plus a very fast kernel
        BatchedGemmIndirect(m, n, k, alphas_device, a_buffer, a_offsets_int, a_ld, b_buffer, b_offsets_int, b_ld,
                            betas_device, c_buffer, c_offsets_int, c_ld, a_do_transpose, b_do_transpose, c_do_transpose,
                            a_conjugate, b_conjugate, a_one, a_two, b_one, b_two, c_one, c_two, batch_count, device, dtype, db);
    }
}

// =================================================================================================

// The indirect version of batched GEMM
void BatchedGemmIndirect(const uint32_t m, const uint32_t n, const uint32_t k, const tart::buffer_ptr& alphas,
                         const tart::buffer_ptr& a_buffer, const std::vector<int>& a_offsets, const size_t a_ld,
                         const tart::buffer_ptr& b_buffer, const std::vector<int>& b_offsets, const size_t b_ld,
                         const tart::buffer_ptr& betas, const tart::buffer_ptr& c_buffer, const std::vector<int>& c_offsets,
                         const size_t c_ld, const bool a_do_transpose, const bool b_do_transpose, const bool c_do_transpose,
                         const bool a_conjugate, const bool b_conjugate, const size_t a_one, const size_t a_two,
                         const size_t b_one, const size_t b_two, const size_t c_one, const size_t c_two, const uint32_t batch_count,
                         tart::device_ptr device, const tart::DType& dtype, const Databases& db)
{
    // Calculates the ceiled versions of m, n, and k
    const auto m_ceiled = Ceil(Ceil(m, db["MWG"]), db["VWM"]);
    const auto n_ceiled = Ceil(Ceil(n, db["NWG"]), db["VWN"]);
    const auto k_ceiled = Ceil(Ceil(k, db["KWG"]), db["VWM"]);

    // Computes the first and second "internal" (ceiled) dimensions
    size_t a_one_i, a_two_i, b_one_i, b_two_i, c_one_i, c_two_i;
    clblast::CalculateInternalDimensions(m, n, k, db["MWG"], db["NWG"], db["KWG"], a_one_i, a_two_i, b_one_i, b_two_i,
                                         c_one_i, c_two_i, db["GEMMK"]);

    // Sets the "internal" offsets
    auto a_offsets_i = std::vector<int>(batch_count);
    auto b_offsets_i = std::vector<int>(batch_count);
    auto c_offsets_i = std::vector<int>(batch_count);
    for (auto batch = size_t{0}; batch < batch_count; ++batch) {
        a_offsets_i[batch] = static_cast<int>(batch * a_one_i * a_two_i);
        b_offsets_i[batch] = static_cast<int>(batch * b_one_i * b_two_i);
        c_offsets_i[batch] = static_cast<int>(batch * c_one_i * c_two_i);
    }

    // Determines whether or not temporary matrices are needed
    auto a_no_temp = a_one == a_one_i && a_two == a_two_i && a_ld == a_one && a_offsets == a_offsets_i &&
                                     !a_do_transpose && !a_conjugate;
    auto b_no_temp = b_one == b_one_i && b_two == b_two_i && b_ld == b_one && b_offsets == b_offsets_i &&
                                     !b_do_transpose && !b_conjugate;
    auto c_no_temp = c_one == c_one_i && c_two == c_two_i && c_ld == c_one && c_offsets == c_offsets_i && !c_do_transpose;

    // Creates the temporary matrices
    const auto a_temp = (a_no_temp) ? a_buffer : device->allocateBuffer(dtype.size()* batch_count * a_one_i * a_two_i);
    const auto b_temp = (b_no_temp) ? b_buffer : device->allocateBuffer(dtype.size()* batch_count * b_one_i * b_two_i);
    const auto c_temp = (c_no_temp) ? c_buffer : device->allocateBuffer(dtype.size()* batch_count * c_one_i * c_two_i);

    // Runs the pre-processing kernel for matrix A
    if (!a_no_temp) {
        auto a_offsets_device = device->allocateBuffer(sizeof(int)*batch_count);
        auto a_offsets_i_device = device->allocateBuffer(batch_count*sizeof(int));
        a_offsets_device->copyIn(a_offsets);
        a_offsets_i_device->copyIn(a_offsets_i);
        PadCopyTransposeMatrixBatched(dtype, db, a_one, a_two, a_ld,
            a_offsets_device, a_buffer, a_one_i, a_two_i, a_one_i, a_offsets_i_device, a_temp,
             true, a_do_transpose, a_conjugate, batch_count);
    }

    // As above, but now for matrix B
    if (!b_no_temp) {
        auto b_offsets_device = device->allocateBuffer(sizeof(int)*batch_count);
        auto b_offsets_i_device = device->allocateBuffer(sizeof(int)*batch_count);
        b_offsets_device->copyIn(b_offsets);
        b_offsets_i_device->copyIn(b_offsets_i);
        PadCopyTransposeMatrixBatched(dtype, db, b_one, b_two, b_ld,
            b_offsets_device, b_buffer, b_one_i, b_two_i, b_one_i, b_offsets_i_device, b_temp,
             true, b_do_transpose, b_conjugate, batch_count);
    }

    // As above, but now for matrix C
    auto c_offsets_device = device->allocateBuffer(sizeof(int)*batch_count);
    auto c_offsets_i_device = device->allocateBuffer(sizeof(int)*batch_count);
    if (!c_no_temp) {
        c_offsets_device->copyIn(c_offsets);
        c_offsets_i_device->copyIn(c_offsets_i);
        PadCopyTransposeMatrixBatched(dtype, db, c_one, c_two, c_ld,
            c_offsets_device, c_buffer, c_one_i, c_two_i, c_one_i, c_offsets_i_device, c_temp,
             true, c_do_transpose, false, batch_count);
    }


    // Retrieves the Xgemm kernel from the compiled binary
    tart::kernel_ptr kernel = getGemmBatched(device, dtype, device->getMetadata().subgroupAdd, db["VWM"], db["VWN"]);
    
    // Sets the kernel arguments
    kernel->setArg(0, static_cast<int>(m_ceiled));
    kernel->setArg(1, static_cast<int>(n_ceiled));
    kernel->setArg(2, static_cast<int>(k_ceiled));
    kernel->setArg(14, alphas);
    kernel->setArg(15, betas);
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
    const auto global = std::vector<size_t>{static_cast<uint32_t>((c_one_i) / db["MWG"]), static_cast<uint32_t>((c_two_i) / db["NWG"]), static_cast<uint32_t>(batch_count)};

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
        PadCopyTransposeMatrixBatched(dtype, db, c_one_i, c_two_i, c_one_i,
                                       c_offsets_i_device, c_temp, c_one, c_two, c_ld, c_offsets_device, c_buffer,
                                       false, c_do_transpose, false, batch_count);
    }
}

// =================================================================================================

// The direct version of batched GEMM
void BatchedGemmDirect(const uint32_t m, const uint32_t n, const uint32_t k, const tart::buffer_ptr& alphas,
                       const tart::buffer_ptr& a_buffer, const std::vector<int>& a_offsets, const size_t a_ld,
                       const tart::buffer_ptr& b_buffer, const std::vector<int>& b_offsets, const size_t b_ld,
                       const tart::buffer_ptr& betas, const tart::buffer_ptr& c_buffer,
                       const std::vector<int>& c_offsets, const size_t c_ld, const bool a_do_transpose,
                       const bool b_do_transpose, const bool c_do_transpose, const bool a_conjugate,
                       const bool b_conjugate, const uint32_t batch_count, tart::device_ptr device, const tart::DType& dtype, const Databases& db)
{
    // Uploads the offsets to the device
    auto a_offsets_device = device->allocateBuffer(sizeof(int) * batch_count);
    auto b_offsets_device = device->allocateBuffer(sizeof(int) * batch_count);
    auto c_offsets_device = device->allocateBuffer(sizeof(int) * batch_count);
    a_offsets_device->copyIn(a_offsets);
    b_offsets_device->copyIn(b_offsets);
    c_offsets_device->copyIn(c_offsets);

    // Retrieves the proper XgemmDirect kernel from the compiled binary
    tart::kernel_ptr kernel = getGemmDirectBatched(device, dtype, db["VWM"], db["VWN"]);

    // Sets the kernel arguments
    kernel->setArg(0, static_cast<int>(m));
    kernel->setArg(1, static_cast<int>(n));
    kernel->setArg(2, static_cast<int>(k));
    
    kernel->setArg(3, a_buffer);
    kernel->setArg(4, static_cast<int>(a_ld));
    
    kernel->setArg(5, b_buffer);
    kernel->setArg(6, static_cast<int>(b_ld));
    kernel->setArg(7, c_buffer);
    kernel->setArg(8, static_cast<int>(c_ld));
    
    kernel->setArg(9, static_cast<int>(c_do_transpose));
    kernel->setArg(10, static_cast<int>(a_conjugate));
    kernel->setArg(11, static_cast<int>(b_conjugate));
    
    // for workaround to no pointer casting allowed
    kernel->setArg(12, a_buffer);
    kernel->setArg(13, b_buffer);
    
    kernel->setArg(14, alphas);
    kernel->setArg(15, betas);
    kernel->setArg(16, a_offsets_device);
    kernel->setArg(17, b_offsets_device);
    kernel->setArg(18, c_offsets_device);

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
