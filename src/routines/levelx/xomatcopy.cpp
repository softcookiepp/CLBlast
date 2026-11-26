
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the functional Xomatcopy routine.
//
// =================================================================================================

#include "routines/levelx/xomatcopy.hpp"

#include <cstddef>
#include <string>
#include <vector>

#include "database/database.hpp"
#include "routines/common.hpp"
#include "utilities/backend.hpp"
#include "utilities/buffer_test.hpp"
#include "utilities/clblast_exceptions.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// The main routine
void DoOmatcopy(const Layout layout, const Transpose a_transpose, const uint32_t m, const uint32_t n,
                const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld,
                const tart::buffer_ptr& b_buffer, const size_t b_offset, const size_t b_ld, const Databases& db,
                tart::device_ptr device, const tart::DType& dtype)
{
    // Makes sure all dimensions are larger than zero
    if ((m == 0) || (n == 0)) {
        throw BLASError(StatusCode::kInvalidDimension);
    }

    // Determines whether to transpose the matrix A
    const auto transpose = (a_transpose != Transpose::kNo);

    // In case of complex data-types, the transpose can also become a conjugate transpose
    const auto conjugate = (a_transpose == Transpose::kConjugate);

    // Computes the dimensions of the two matrices
    const auto rotated = (layout == Layout::kRowMajor);
    const auto a_one = (rotated) ? n : m;
    const auto a_two = (rotated) ? m : n;
    const auto b_one = (transpose) ? a_two : a_one;
    const auto b_two = (transpose) ? a_one : a_two;

    // Tests the matrices for validity
    TestMatrixA(dtype, a_one, a_two, a_buffer, a_offset, a_ld);
    TestMatrixB(dtype, b_one, b_two, b_buffer, b_offset, b_ld);

    PadCopyTransposeMatrix(dtype, db, a_one, a_two, a_ld, a_offset, a_buffer, b_one,
                            b_two, b_ld, b_offset, b_buffer, alpha, false, transpose, conjugate);
}

// =================================================================================================
}	// namespace clblast
