
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the functional Xherk routine.
//
// =================================================================================================

#include "routines/level3/xherk.hpp"

#include <cstddef>
#include <string>
#include <vector>

#include "database/database.hpp"
#include "routine_aot_cache.hpp"
#include "routines/common.hpp"
#include "routines/level3/xgemm.hpp"
#include "utilities/backend.hpp"
#include "utilities/buffer_test.hpp"
#include "utilities/utilities.hpp"
#include "xtrmm.hpp"

namespace clblast {
// =================================================================================================

// The main routine
void DoHerk(const Layout layout, const Triangle triangle, const Transpose a_transpose, const uint32_t n,
            const uint32_t k, const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer, const size_t a_offset,
            const size_t a_ld, const tart::Scalar& beta, const tart::buffer_ptr& c_buffer, const size_t c_offset,
            const size_t c_ld, const Databases& db,
            tart::device_ptr device, const tart::DType& dtype) {
    const auto b_transpose = (a_transpose != Transpose::kNo) ? Transpose::kNo : Transpose::kYes;
    const auto& b_buffer = a_buffer;
    const auto b_offset = a_offset;
    const auto b_ld = a_ld;
    HerkAB(layout, triangle, a_transpose, b_transpose, n, k, alpha, a_buffer, a_offset, a_ld, b_buffer, b_offset,
           b_ld, beta, c_buffer, c_offset, c_ld, true, db, device, dtype);
}

void HerkAB(const Layout layout, const Triangle triangle, const Transpose a_transpose,
            const Transpose b_transpose, const uint32_t n, const uint32_t k, const tart::Scalar& complex_alpha,
            const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld, const tart::buffer_ptr& b_buffer,
            const size_t b_offset, const size_t b_ld, const tart::Scalar& complex_beta, const tart::buffer_ptr& c_buffer,
            const size_t c_offset, const size_t c_ld,
            const bool diagonal_to_zero, const Databases& db,
            tart::device_ptr device, const tart::DType& dtype)
{
    // Computes the transpose/conjugate options and sets the a/b/c sizes
    bool a_do_transpose, b_do_transpose, c_do_transpose, dummy1, dummy2;
    size_t a_one, a_two, b_one, b_two, c_one, c_two;
    ProcessArguments(layout, a_transpose, b_transpose, n, n, k, a_one, a_two, b_one, b_two, c_one, c_two,
                     a_do_transpose, b_do_transpose, c_do_transpose, dummy1, dummy2, db["GEMMK"]);

    // Determines whether to apply the conjugate transpose to matrix B or matrix A
    auto a_conjugate = (a_transpose != Transpose::kNo);
    auto b_conjugate = (b_transpose != Transpose::kNo);

    // Tests the two matrices (A, C) for validity
    TestMatrixA(dtype, a_one, a_two, a_buffer, a_offset, a_ld);
    TestMatrixB(dtype, b_one, b_two, b_buffer, b_offset, b_ld);
    TestMatrixC(dtype, n, n, c_buffer, c_offset, c_ld);

    // Calculates the ceiled versions of n and k
    auto n_ceiled = Ceil(Ceil(n, db["MWG"]), db["NWG"]);
    auto k_ceiled = Ceil(k, db["KWG"] * db["KREG"]);

    // Computes the first and second "internal" (ceiled) dimensions
    const auto a_one_i = (a_want_rotated_(db["GEMMK"])) ? k_ceiled : n_ceiled;
    const auto a_two_i = (a_want_rotated_(db["GEMMK"])) ? n_ceiled : k_ceiled;
    const auto b_one_i = (!b_want_rotated_(db["GEMMK"])) ? k_ceiled : n_ceiled;
    const auto b_two_i = (!b_want_rotated_(db["GEMMK"])) ? n_ceiled : k_ceiled;

    // Determines whether or not temporary matrices are needed
    const auto a_no_temp = NoTempBuffer(a_one, a_one_i, a_two, a_two_i, a_ld, a_offset, a_do_transpose, a_conjugate);
    const auto b_no_temp = NoTempBuffer(b_one, b_one_i, b_two, b_two_i, b_ld, b_offset, b_do_transpose, b_conjugate);

    // Creates the temporary matrices
    auto a_temp = (a_no_temp) ? a_buffer : device->allocateBuffer(dtype.size() * a_one_i * a_two_i);
    auto b_temp = (b_no_temp) ? b_buffer : device->allocateBuffer(dtype.size() * b_one_i * b_two_i);
    auto c_temp = device->allocateBuffer(dtype.size() * n_ceiled * n_ceiled);

    // Runs the pre-processing kernel for matrix A
    if (!a_no_temp) {
        PadCopyTransposeMatrix(dtype, db, a_one, a_two, a_ld, a_offset,
                                 a_buffer, a_one_i, a_two_i, a_one_i, 0, a_temp, tart::Scalar::one(), true,
                                 a_do_transpose, a_conjugate,
                                 false, false, false,
                                 false, true, false, false);
    }
    if (!b_no_temp) {
        PadCopyTransposeMatrix(dtype, db, b_one, b_two, b_ld, b_offset,
                                 b_buffer, b_one_i, b_two_i, b_one_i, 0, b_temp, tart::Scalar::one(), true,
                                 b_do_transpose, b_conjugate,
                                 false, false, false,
                                 false, true, false, false);
    }

    // Creates a (possibly padded) copy of matrix C
    PadCopyTransposeMatrix(dtype, db, n, n, c_ld, c_offset, c_buffer,
                            n_ceiled, n_ceiled, n_ceiled, 0, c_temp, tart::Scalar::one(), true, c_do_transpose,
                            false,
                            false, false, false,
                            false, true, false, false);

    // Retrieves the XgemmUpper or XgemmLower kernel
    tart::kernel_ptr kernel = (triangle == Triangle::kUpper) ?
        getGemmUpper(device, dtype, device->getMetadata().subgroupAdd, db["VWM"], db["VWN"]) :
        getGemmLower(device, dtype, device->getMetadata().subgroupAdd, db["VWM"], db["VWN"]);
    
    // Sets the kernel arguments
    kernel->setArg(0, static_cast<int>(n_ceiled));
    kernel->setArg(1, static_cast<int>(k_ceiled));
    kernel->setArgAs(dtype, 2, (complex_alpha));
    kernel->setArgAs(dtype, 3, (complex_beta));
    kernel->setArg(4, a_temp);
    kernel->setArg(5, b_temp);
    kernel->setArg(6, c_temp);
    kernel->setArg(7, a_temp);
    kernel->setArg(8, b_temp);

    // Computes the global and local thread sizes
    auto global = std::vector<uint32_t>{static_cast<uint32_t>((n_ceiled) / db["MWG"]), static_cast<uint32_t>((n_ceiled) / db["NWG"])};
    auto local = std::vector<uint32_t>{db["MDIMC"], db["NDIMC"]};

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

    // Runs the post-processing kernel
    const auto upper =
            c_want_rotated_(db["GEMMK"]) ? (triangle == Triangle::kLower) : (triangle == Triangle::kUpper);
    const auto lower = !upper;
    PadCopyTransposeMatrix(dtype, db, n_ceiled, n_ceiled, n_ceiled, 0, c_temp, n,
                            n, c_ld, c_offset, c_buffer, tart::Scalar::one(), false, c_do_transpose, false, upper,
                            lower, diagonal_to_zero,
                            false, true, false, false);
}

// Xherk class implementation
Xherk::Xherk(const tart::DType& dtype, [[maybe_unused]] Queue& queue, const std::string& name) {}

void Xherk::DoHerk(const Layout layout, const Triangle triangle, const Transpose a_transpose, const uint32_t n,
                   const uint32_t k, const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer, const size_t a_offset,
                   const size_t a_ld, const tart::Scalar& beta, const tart::buffer_ptr& c_buffer, const size_t c_offset,
                   const size_t c_ld, const Databases& db,
                   tart::device_ptr device, const tart::DType& dtype) {
    DoHerk(layout, triangle, a_transpose, n, k, alpha, a_buffer, a_offset, a_ld, beta, c_buffer, c_offset, c_ld,
           db, device ? device : device, dtype != tart::dtypes::unknown_type ? dtype : dtype);
}

void Xherk::HerkAB(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Transpose b_transpose,
                   const uint32_t n, const uint32_t k, const tart::Scalar& complex_alpha, const tart::buffer_ptr& a_buffer,
                   const size_t a_offset, const size_t a_ld, const tart::buffer_ptr& b_buffer, const size_t b_offset,
                   const size_t b_ld, const tart::Scalar& complex_beta, const tart::buffer_ptr& c_buffer,
                   const size_t c_offset, const size_t c_ld, const bool diagonal_to_zero, const Databases& db,
                   tart::device_ptr device, const tart::DType& dtype) {
    HerkAB(layout, triangle, a_transpose, b_transpose, n, k, complex_alpha, a_buffer, a_offset, a_ld, b_buffer, b_offset,
           b_ld, complex_beta, c_buffer, c_offset, c_ld, diagonal_to_zero, db,
           device ? device : device, dtype != tart::dtypes::unknown_type ? dtype : dtype);
}

// =================================================================================================
}  // namespace clblast
