
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the functional interface for the Xomatcopy routine.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XOMATCOPY_H_
#define CLBLAST_ROUTINES_XOMATCOPY_H_

#include <cstddef>
#include <string>

#include "database/database.hpp"
#include "utilities/backend.hpp"
#include "tart/tart.hpp"

namespace clblast {
// =================================================================================================

// Functional implementation of Xomatcopy (out-of-place matrix transpose/copy)
void DoOmatcopy(const Layout layout, const Transpose a_transpose, const size_t m, const size_t n,
                const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld,
                const tart::buffer_ptr& b_buffer, const size_t b_offset, const size_t b_ld, const Databases& db,
                tart::device_ptr device, const tart::DType& dtype);

// =================================================================================================
}	// namespace clblast

// CLBLAST_ROUTINES_XOMATCOPY_H_
#endif
