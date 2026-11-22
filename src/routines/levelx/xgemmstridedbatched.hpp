
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the functional interface for the XgemmStridedBatched routine.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XGEMMSTRIDEDBATCHED_H_
#define CLBLAST_ROUTINES_XGEMMSTRIDEDBATCHED_H_

#include <cstddef>
#include <string>

#include "database/database.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// Functional implementation of XgemmStridedBatched
void DoGemmStridedBatched(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                          const size_t m, const size_t n, const size_t k, const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer,
                          const size_t a_offset, const size_t a_ld, const size_t a_stride, const tart::buffer_ptr& b_buffer,
                          const size_t b_offset, const size_t b_ld, const size_t b_stride, const tart::Scalar& beta,
                          const tart::buffer_ptr& c_buffer, const size_t c_offset, const size_t c_ld, const size_t c_stride,
                          const size_t batch_count, tart::device_ptr device, const tart::DType& dtype, const Databases& db);

// Indirect version of strided batched GEMM
void BatchedGemmIndirect(const size_t m, const size_t n, const size_t k, const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer,
                         const size_t a_offset, const size_t a_ld, const size_t a_stride, const tart::buffer_ptr& b_buffer,
                         const size_t b_offset, const size_t b_ld, const size_t b_stride, const tart::Scalar& beta,
                         const tart::buffer_ptr& c_buffer, const size_t c_offset, const size_t c_ld, const size_t c_stride,
                         const bool a_do_transpose, const bool b_do_transpose, const bool c_do_transpose,
                         const bool a_conjugate, const bool b_conjugate, const size_t a_one, const size_t a_two,
                         const size_t b_one, const size_t b_two, const size_t c_one, const size_t c_two,
                         const size_t batch_count, tart::device_ptr device, const tart::DType& dtype, const Databases& db);

// Direct version of strided batched GEMM
void BatchedGemmDirect(const size_t m, const size_t n, const size_t k, const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer,
                       const size_t a_offset, const size_t a_ld, const size_t a_stride, const tart::buffer_ptr& b_buffer,
                       const size_t b_offset, const size_t b_ld, const size_t b_stride, const tart::Scalar& beta,
                       const tart::buffer_ptr& c_buffer, const size_t c_offset, const size_t c_ld, const size_t c_stride,
                       const bool a_do_transpose, const bool b_do_transpose, const bool c_do_transpose,
                       const bool a_conjugate, const bool b_conjugate, const size_t batch_count,
                       tart::device_ptr device, const tart::DType& dtype, const Databases& db);

// =================================================================================================
}	// namespace clblast

// CLBLAST_ROUTINES_XGEMMSTRIDEDBATCHED_H_
#endif
