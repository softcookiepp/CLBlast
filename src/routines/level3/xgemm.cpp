
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the functional Xgemm routine.
//
// =================================================================================================

#include "routines/level3/xgemm.hpp"
#include "routines/level2/xgemv.hpp"

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

// Computes the sizes and offsets for (optional) temporary buffers for the 3 matrices
size_t ComputeTempSize(const bool a_no_temp, const bool b_no_temp, const bool c_no_temp, const size_t a_size,
                       const size_t b_size, const size_t c_size, size_t& b_temp_offset, size_t& c_temp_offset) {
    auto temp_size = size_t{0};
    if (!a_no_temp) {
        temp_size += a_size;
    }
    if (!b_no_temp) {
        b_temp_offset = temp_size;
        temp_size += b_size;
    }
    if (!c_no_temp) {
        c_temp_offset = temp_size;
        temp_size += c_size;
    }
    return temp_size;
}


// Computes the first and second "internal" (ceiled) dimensions of the 3 matrices
void CalculateInternalDimensions(const uint32_t m, const uint32_t n, const uint32_t k, const size_t mwg,
                                 const size_t nwg, const size_t kwg, size_t& a_one_i, size_t& a_two_i,
                                 size_t& b_one_i, size_t& b_two_i, size_t& c_one_i, size_t& c_two_i,
                                 const size_t gemm_kernel_id) {
    const auto global_divider_one = c_want_rotated_(gemm_kernel_id) ? nwg : mwg;
    const auto global_divider_two = c_want_rotated_(gemm_kernel_id) ? mwg : nwg;
    const auto m_ceiled = Ceil(m, global_divider_one);
    const auto n_ceiled = Ceil(n, global_divider_two);
    const auto k_ceiled = Ceil(k, kwg);
    a_one_i = (a_want_rotated_(gemm_kernel_id)) ? k_ceiled : m_ceiled;
    a_two_i = (a_want_rotated_(gemm_kernel_id)) ? m_ceiled : k_ceiled;
    b_one_i = (b_want_rotated_(gemm_kernel_id)) ? n_ceiled : k_ceiled;
    b_two_i = (b_want_rotated_(gemm_kernel_id)) ? k_ceiled : n_ceiled;
    c_one_i = (c_want_rotated_(gemm_kernel_id)) ? n_ceiled : m_ceiled;
    c_two_i = (c_want_rotated_(gemm_kernel_id)) ? m_ceiled : n_ceiled;
}

// Selects which version of GEMM to run
bool UseDirectKernel(const uint32_t m, const uint32_t n, const uint32_t k, const size_t min_indirect_size) {
    const auto m_n_k =
            static_cast<unsigned long long>(m) * static_cast<unsigned long long>(n) * static_cast<unsigned long long>(k);
    const auto min_indirect_size_ll = static_cast<unsigned long long>(min_indirect_size);
    const auto min_indirect_size_e3 = min_indirect_size_ll * min_indirect_size_ll * min_indirect_size_ll;
    return (m_n_k < min_indirect_size_e3);
}

// Process the user-arguments, computes secondary parameters
void ProcessArguments(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                      const uint32_t m, const uint32_t n, const uint32_t k, size_t& a_one, size_t& a_two,
                      size_t& b_one, size_t& b_two, size_t& c_one, size_t& c_two, bool& a_do_transpose,
                      bool& b_do_transpose, bool& c_do_transpose, bool& a_conjugate, bool& b_conjugate,
                      const size_t gemm_kernel_id) {
    // Makes sure all dimensions are larger than zero
    if ((m == 0) || (n == 0) || (k == 0)) {
        throw BLASError(StatusCode::kInvalidDimension);
    }

    // Computes whether or not the matrices are transposed in memory.
    const auto a_rotated = (layout == Layout::kColMajor && a_transpose != Transpose::kNo) ||
                           (layout == Layout::kRowMajor && a_transpose == Transpose::kNo);
    const auto b_rotated = (layout == Layout::kColMajor && b_transpose != Transpose::kNo) ||
                           (layout == Layout::kRowMajor && b_transpose == Transpose::kNo);
    const auto c_rotated = (layout == Layout::kRowMajor);
    a_do_transpose = a_rotated != a_want_rotated_(gemm_kernel_id);
    b_do_transpose = b_rotated != b_want_rotated_(gemm_kernel_id);
    c_do_transpose = c_rotated != c_want_rotated_(gemm_kernel_id);

    // In case of complex data-types, the transpose can also become a conjugate transpose
    a_conjugate = (a_transpose == Transpose::kConjugate);
    b_conjugate = (b_transpose == Transpose::kConjugate);

    // Computes the first and second dimensions of the 3 matrices
    a_one = (a_rotated) ? k : m;
    a_two = (a_rotated) ? m : k;
    b_one = (b_rotated) ? n : k;
    b_two = (b_rotated) ? k : n;
    c_one = (c_rotated) ? n : m;
    c_two = (c_rotated) ? m : n;
}

// Computes the size of the temporary GEMM buffer
size_t GetTempSize(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                   const uint32_t m, const uint32_t n, const uint32_t k, const size_t a_offset, const size_t a_ld,
                   const size_t b_offset, const size_t b_ld, const size_t c_offset, const size_t c_ld,
                   const size_t mwg, const size_t nwg, const size_t kwg, const size_t gemm_kernel_id) {
    size_t a_one, a_two, b_one, b_two, c_one, c_two;
    bool a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate;
    ProcessArguments(layout, a_transpose, b_transpose, m, n, k, a_one, a_two, b_one, b_two, c_one, c_two,
                     a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate, gemm_kernel_id);

    size_t a_one_i, a_two_i, b_one_i, b_two_i, c_one_i, c_two_i;
    CalculateInternalDimensions(m, n, k, mwg, nwg, kwg, a_one_i, a_two_i, b_one_i, b_two_i, c_one_i, c_two_i, gemm_kernel_id);

    auto a_no_temp = NoTempBuffer(a_one, a_one_i, a_two, a_two_i, a_ld, a_offset, a_do_transpose, a_conjugate);
    auto b_no_temp = NoTempBuffer(b_one, b_one_i, b_two, b_two_i, b_ld, b_offset, b_do_transpose, b_conjugate);
    auto c_no_temp = NoTempBuffer(c_one, c_one_i, c_two, c_two_i, c_ld, c_offset, c_do_transpose, false);

    auto b_temp_offset = size_t{0};
    auto c_temp_offset = size_t{0};
    return ComputeTempSize(a_no_temp, b_no_temp, c_no_temp, a_one_i * a_two_i, b_one_i * b_two_i,
                           c_one_i * c_two_i, b_temp_offset, c_temp_offset);
}

// =================================================================================================

// The main routine
void DoGemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose, const uint32_t m,
            const uint32_t n, const uint32_t k, const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer, const size_t a_offset,
            const size_t a_ld, const tart::buffer_ptr& b_buffer, const size_t b_offset, const size_t b_ld,
            const tart::Scalar& beta, const tart::buffer_ptr& c_buffer, const size_t c_offset, const size_t c_ld,
            const tart::buffer_ptr& temp_buffer, const bool temp_buffer_provided, const Databases& db,
            tart::device_ptr device, const tart::DType& dtype) {

    // Two methods to choose from, select which one to run
    const auto do_gemm_direct = UseDirectKernel(m, n, k, db["XGEMM_MIN_INDIRECT_SIZE"]);
    const auto gemm_kernel_id = (do_gemm_direct) ? 0 : db["GEMMK"];

    // Computes the transpose/conjugate options and sets the a/b/c sizes based on that
    bool a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate;
    size_t a_one, a_two, b_one, b_two, c_one, c_two;
    ProcessArguments(layout, a_transpose, b_transpose, m, n, k, a_one, a_two, b_one, b_two, c_one, c_two, a_do_transpose,
                     b_do_transpose, c_do_transpose, a_conjugate, b_conjugate, gemm_kernel_id);

    // Tests three matrices (A, B, C) for validity
    TestMatrixA(dtype, a_one, a_two, a_buffer, a_offset, a_ld);
    TestMatrixB(dtype, b_one, b_two, b_buffer, b_offset, b_ld);
    TestMatrixC(dtype, c_one, c_two, c_buffer, c_offset, c_ld);

    // Selects which version of GEMM to run
    if (do_gemm_direct) {	// for small sizes (single kernel)
        GemmDirect(m, n, k, alpha, a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, beta, c_buffer, c_offset, c_ld,
                   a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate, db, device, dtype);
    } else {	// for larger sizes (pre/post-processing plus a very fast kernel)
        GemmIndirect(m, n, k, alpha, a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, beta, c_buffer, c_offset, c_ld,
                     a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate, a_one, a_two, b_one, b_two,
                     c_one, c_two, temp_buffer, temp_buffer_provided, db, device, dtype);
    }
}

// =================================================================================================

// The indirect version of GEMM
void GemmIndirect(const uint32_t m, const uint32_t n, const uint32_t k, const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer,
                  const size_t a_offset, const size_t a_ld, const tart::buffer_ptr& b_buffer, const size_t b_offset,
                  const size_t b_ld, const tart::Scalar& beta, const tart::buffer_ptr& c_buffer, const size_t c_offset,
                  const size_t c_ld, const bool a_do_transpose, const bool b_do_transpose,
                  const bool c_do_transpose, const bool a_conjugate, const bool b_conjugate,
                  const size_t a_one, const size_t a_two, const size_t b_one, const size_t b_two,
                  const size_t c_one, const size_t c_two, const tart::buffer_ptr& temp_buffer,
                  const bool temp_buffer_provided, const Databases& db,
                  tart::device_ptr device, const tart::DType& dtype)
{
    // Calculates the ceiled versions of m, n, and k
    const auto global_divider_one = c_want_rotated_(db["GEMMK"]) ? db["NWG"] : db["MWG"];
    const auto global_divider_two = c_want_rotated_(db["GEMMK"]) ? db["MWG"] : db["NWG"];
    const auto m_ceiled = Ceil(m, global_divider_one);
    const auto n_ceiled = Ceil(n, global_divider_two);
    const auto k_ceiled = Ceil(k, db["KWG"] * db["KREG"]);

    // Computes the first and second "internal" (ceiled) dimensions
    size_t a_one_i, a_two_i, b_one_i, b_two_i, c_one_i, c_two_i;
    CalculateInternalDimensions(m, n, k, db["MWG"], db["NWG"], db["KWG"] * db["KREG"], a_one_i, a_two_i, b_one_i,
                                b_two_i, c_one_i, c_two_i, db["GEMMK"]);

    // Determines whether or not temporary matrices are needed
    auto a_no_temp = NoTempBuffer(a_one, a_one_i, a_two, a_two_i, a_ld, a_offset, a_do_transpose, a_conjugate);
    auto b_no_temp = NoTempBuffer(b_one, b_one_i, b_two, b_two_i, b_ld, b_offset, b_do_transpose, b_conjugate);
    auto c_no_temp = NoTempBuffer(c_one, c_one_i, c_two, c_two_i, c_ld, c_offset, c_do_transpose, false);

    // Computes the sizes and offsets for (optional) temporary buffers
    auto b_temp_offset = size_t{0};
    auto c_temp_offset = size_t{0};
    const auto temp_size = ComputeTempSize(a_no_temp, b_no_temp, c_no_temp, a_one_i * a_two_i, b_one_i * b_two_i,
                                           c_one_i * c_two_i, b_temp_offset, c_temp_offset);
    if (!IsMultiple(b_temp_offset, db["VWN"])) {
        throw BLASError(StatusCode::kUnexpectedError);
    }
    if (!IsMultiple(c_temp_offset, db["VWM"])) {
        throw BLASError(StatusCode::kUnexpectedError);
    }

    // Creates the buffer for the (optional) temporary matrices
    auto temp_buffer_all =
            (temp_buffer_provided) ? temp_buffer : ((temp_size > 0) ? device->allocateBuffer(temp_size*dtype.size()) : a_buffer);
    // Verifies if the provided temporary buffer is large enough
    if (temp_buffer_provided) {
        const auto required_size = temp_size * dtype.size();
        if (temp_buffer_all->getSize() < required_size) {
            // temporary hack due to indirect kernel being forced on under certain circumstances
            if (temp_size == 0) throw std::runtime_error("this probably shouldn't happen");
            temp_buffer_all = device->allocateBuffer(temp_size);
        }
    }

    // Sets the buffer pointers for (temp) matrices A, B, and C
    const auto a_temp = (a_no_temp) ? a_buffer : temp_buffer_all;
    const auto b_temp = (b_no_temp) ? b_buffer : temp_buffer_all;
    const auto c_temp = (c_no_temp) ? c_buffer : temp_buffer_all;

    // Runs the pre-processing kernel for matrix A
    if (!a_no_temp)
    {
        PadCopyTransposeMatrix(dtype, db, a_one, a_two, a_ld, a_offset,
                                 a_buffer, a_one_i, a_two_i, a_one_i, 0, a_temp, tart::Scalar::one(), true,
                                 a_do_transpose, a_conjugate,
                                 false, false, false);
    }

    // As above, but now for matrix B
    if (!b_no_temp)
    {
        PadCopyTransposeMatrix(dtype, db, b_one, b_two, b_ld, b_offset,
                                 b_buffer, b_one_i, b_two_i, b_one_i, b_temp_offset, b_temp, tart::Scalar::one(), true,
                                 b_do_transpose, b_conjugate,
                                 false, false, false);
    }

    // As above, but now for matrix C
    if (!c_no_temp && beta != tart::Scalar::zero())
    {
        PadCopyTransposeMatrix(dtype, db, c_one, c_two, c_ld, c_offset,
                                 c_buffer, c_one_i, c_two_i, c_one_i, c_temp_offset, c_temp, tart::Scalar::one(), true,
                                 c_do_transpose, false,
                                 false, false, false);
    }

    // Retrieves the Xgemm kernel from the compiled binary
    tart::kernel_ptr kernel = getGemm(device, dtype, device->getMetadata().subgroupAdd, db["VWM"], db["VWN"]);

    // Sets the kernel arguments
    kernel->setArg(0, static_cast<int>(m_ceiled));
    kernel->setArg(1, static_cast<int>(n_ceiled));
    kernel->setArg(2, static_cast<int>(k_ceiled));
    kernel->setArgAs(dtype, 3, alpha);
    kernel->setArgAs(dtype, 4, beta);
    kernel->setArg(5, a_temp);
    kernel->setArg(6, b_temp);
    kernel->setArg(7, c_temp);
    kernel->setArg(8, a_temp);
    kernel->setArg(9, b_temp);
    kernel->setArg(10, static_cast<int>(b_temp_offset / db["VWN"]));
    kernel->setArg(11, static_cast<int>(c_temp_offset / db["VWM"]));

    // Computes the global and local thread sizes
    const auto global = std::vector<uint32_t>{static_cast<uint32_t>(c_one_i / db["MWG"]), static_cast<uint32_t>(c_two_i / db["NWG"])};

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
        PadCopyTransposeMatrix(dtype, db, c_one_i, c_two_i, c_one_i, c_temp_offset,
                                 c_temp, c_one, c_two, c_ld, c_offset, c_buffer, tart::Scalar::one(), false,
                                 c_do_transpose, false, false, false, false);
    }
}

// =================================================================================================

// The direct version of GEMM
void GemmDirect(const uint32_t m, const uint32_t n, const uint32_t k, const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer,
                const size_t a_offset, const size_t a_ld, const tart::buffer_ptr& b_buffer, const size_t b_offset,
                const size_t b_ld, const tart::Scalar& beta, const tart::buffer_ptr& c_buffer, const size_t c_offset,
                const size_t c_ld, const bool a_do_transpose, const bool b_do_transpose, const bool c_do_transpose,
                const bool a_conjugate, const bool b_conjugate, const Databases& db,
                tart::device_ptr device, const tart::DType& dtype)
{
    // Retrieves the XgemmDirect kernel from the compiled binary
    tart::kernel_ptr kernel = getGemmDirect(device, dtype, db["VWM"], db["VWN"]);

    // Sets the kernel arguments
    kernel->setArg(0, static_cast<int>(m));
    kernel->setArg(1, static_cast<int>(n));
    kernel->setArg(2, static_cast<int>(k));
    kernel->setArgAs(dtype, 3, alpha);
    kernel->setArgAs(dtype, 4, beta);
    kernel->setArg(5, a_buffer);
    kernel->setArg(6, static_cast<int>(a_offset));
    kernel->setArg(7, static_cast<int>(a_ld));
    kernel->setArg(8, b_buffer);
    kernel->setArg(9, static_cast<int>(b_offset));
    kernel->setArg(10, static_cast<int>(b_ld));
    kernel->setArg(11, c_buffer);
    kernel->setArg(12, static_cast<int>(c_offset));
    kernel->setArg(13, static_cast<int>(c_ld));
    kernel->setArg(14, static_cast<int>(c_do_transpose));
    kernel->setArg(15, static_cast<int>(a_conjugate));
    kernel->setArg(16, static_cast<int>(b_conjugate));
    
    // provide the same buffers twice to get around GLSL's lack of pointer casting
    kernel->setArg(17, a_buffer);
    kernel->setArg(18, b_buffer);

    // Computes the global and local thread sizes
    const auto m_ceiled = Ceil(m, db["WGD"]);
    const auto n_ceiled = Ceil(n, db["WGD"]);
    const auto global = std::vector<uint32_t>{static_cast<uint32_t>(m_ceiled / db["WGD"]), static_cast<uint32_t>(n_ceiled / db["WGD"]), 1};
    
    const std::vector<uint32_t> spec({
        static_cast<uint32_t>(db["WGD"]),
        static_cast<uint32_t>(db["MDIMCD"]),
        static_cast<uint32_t>(db["NDIMCD"]),
        static_cast<uint32_t>(db["MDIMAD"]),
        static_cast<uint32_t>(db["NDIMBD"]),
        static_cast<uint32_t>(db["KWID"]),
        static_cast<uint32_t>(db["PADA"]),
        static_cast<uint32_t>(db["PADB"]),
        static_cast<uint32_t>(a_do_transpose),
        static_cast<uint32_t>(b_do_transpose)
    });

    // Launches the kernel
    kernel->enqueue(global, spec);
}

// Xgemm class implementation
Xgemm::Xgemm(const tart::DType& dtype, [[maybe_unused]] Queue& queue, const std::string& name) {}

void Xgemm::DoGemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose, const uint32_t m,
                   const uint32_t n, const uint32_t k, const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer, const size_t a_offset,
                   const size_t a_ld, const tart::buffer_ptr& b_buffer, const size_t b_offset, const size_t b_ld,
                   const tart::Scalar& beta, const tart::buffer_ptr& c_buffer, const size_t c_offset, const size_t c_ld,
                   const tart::buffer_ptr& temp_buffer, const bool temp_buffer_provided, const Databases& db,
                   tart::device_ptr device, const tart::DType& dtype) {
    DoGemm(layout, a_transpose, b_transpose, m, n, k, alpha, a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld,
           beta, c_buffer, c_offset, c_ld, temp_buffer, temp_buffer_provided, db,
           device ? device : device, dtype != tart::dtypes::unknown_type ? dtype : dtype);
}

void Xgemm::GemmIndirect(const uint32_t m, const uint32_t n, const uint32_t k, const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer,
                         const size_t a_offset, const size_t a_ld, const tart::buffer_ptr& b_buffer, const size_t b_offset,
                         const size_t b_ld, const tart::Scalar& beta, const tart::buffer_ptr& c_buffer, const size_t c_offset,
                         const size_t c_ld, const bool a_do_transpose, const bool b_do_transpose, const bool c_do_transpose,
                         const bool a_conjugate, const bool b_conjugate, const size_t a_one, const size_t a_two,
                         const size_t b_one, const size_t b_two, const size_t c_one, const size_t c_two,
                         const tart::buffer_ptr& temp_buffer, const bool temp_buffer_provided, const Databases& db,
                         tart::device_ptr device, const tart::DType& dtype) {
    GemmIndirect(m, n, k, alpha, a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, beta, c_buffer, c_offset, c_ld,
                 a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate, a_one, a_two, b_one, b_two,
                 c_one, c_two, temp_buffer, temp_buffer_provided, db,
                 device ? device : device, dtype != tart::dtypes::unknown_type ? dtype : dtype);
}

void Xgemm::GemmDirect(const uint32_t m, const uint32_t n, const uint32_t k, const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer,
                       const size_t a_offset, const size_t a_ld, const tart::buffer_ptr& b_buffer, const size_t b_offset,
                       const size_t b_ld, const tart::Scalar& beta, const tart::buffer_ptr& c_buffer, const size_t c_offset,
                       const size_t c_ld, const bool a_do_transpose, const bool b_do_transpose, const bool c_do_transpose,
                       const bool a_conjugate, const bool b_conjugate, const Databases& db,
                       tart::device_ptr device, const tart::DType& dtype) {
    GemmDirect(m, n, k, alpha, a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, beta, c_buffer, c_offset, c_ld,
               a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate, db,
               device ? device : device, dtype != tart::dtypes::unknown_type ? dtype : dtype);
}

// Xgemv class implementation
Xgemv::Xgemv(const tart::DType& dtype, [[maybe_unused]] Queue& queue, const std::string& name) {}

void Xgemv::DoGemv(const Layout layout, const Transpose a_transpose, const uint32_t m, const uint32_t n,
                   const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld,
                   const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
                   const tart::Scalar& beta, const tart::buffer_ptr& y_buffer, const size_t y_offset, const size_t y_inc,
                   const Databases& db,
                   tart::device_ptr device, const tart::DType& dtype) {
    DoGemv(layout, a_transpose, m, n, alpha, a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc,
           beta, y_buffer, y_offset, y_inc, db,
           device ? device : device, dtype != tart::dtypes::unknown_type ? dtype : dtype);
}

void Xgemv::MatVec(const Layout layout, const Transpose a_transpose, const uint32_t m, const uint32_t n,
                   const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld,
                   const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
                   const tart::Scalar& beta, const tart::buffer_ptr& y_buffer, const size_t y_offset, const size_t y_inc,
                   bool fast_kernel, bool fast_kernel_rot, const size_t parameter, const bool packed,
                   const size_t kl, const size_t ku, const std::string& name, [[maybe_unused]] const Databases& db,
                   tart::device_ptr device, const tart::DType& dtype) {
    MatVec(layout, a_transpose, m, n, alpha, a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc,
           beta, y_buffer, y_offset, y_inc, fast_kernel, fast_kernel_rot, parameter, packed, kl, ku, name,
           db, device ? device : device, dtype != tart::dtypes::unknown_type ? dtype : dtype);
}

// =================================================================================================
}	// namespace clblast
