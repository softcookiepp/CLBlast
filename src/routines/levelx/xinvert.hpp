
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the functional interface for the Xinvert routine.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XINVERT_H_
#define CLBLAST_ROUTINES_XINVERT_H_

#include <cstddef>
#include <string>

#include "database/database.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// Functional implementation of Xinvert (inverts diagonal square blocks of a matrix)
void InvertMatrixDiagonalBlocks(const Layout layout, const Triangle triangle, const Diagonal diag, const uint32_t n,
                                const uint32_t block_size, const tart::buffer_ptr& src, const uint32_t offset,
                                const size_t ld_src, tart::buffer_ptr& dest, const tart::device_ptr& device,
                                const tart::DType& dtype, const Databases& db);

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XINVERT_H_
#endif
