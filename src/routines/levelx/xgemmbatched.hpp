
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the functional interface for the XgemmBatched routine.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XGEMMBATCHED_H_
#define CLBLAST_ROUTINES_XGEMMBATCHED_H_

#include <cstddef>
#include <string>
#include <vector>

#include "database/database.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// Functional implementation of XgemmBatched
void DoGemmBatched(const Layout layout, const Transpose a_transpose, const Transpose b_transpose, const size_t m,
                   const size_t n, const size_t k, const tart::buffer_ptr& alphas_device, const tart::buffer_ptr& a_buffer,
                   const std::vector<size_t>& a_offsets, const size_t a_ld, const tart::buffer_ptr& b_buffer,
                   const std::vector<size_t>& b_offsets, const size_t b_ld, const tart::buffer_ptr& betas_device,
                   const tart::buffer_ptr& c_buffer, const std::vector<size_t>& c_offsets, const size_t c_ld,
                   const size_t batch_count, tart::device_ptr device, const tart::DType& dtype, const Databases& db);

// Helper function for template method
template <typename T>
void UploadGemmBatched(const size_t batch_count, const std::vector<T>& alphas, const std::vector<T>& betas,
                       const tart::buffer_ptr& a_buffer, const std::vector<size_t>& a_offsets, const size_t a_ld,
                       const tart::buffer_ptr& b_buffer, const std::vector<size_t>& b_offsets, const size_t b_ld,
                       const tart::buffer_ptr& c_buffer, const std::vector<size_t>& c_offsets, const size_t c_ld,
                       const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                       const size_t m, const size_t n, const size_t k, tart::device_ptr device, const tart::DType& dtype,
                       const Databases& db);

// Indirect version of batched GEMM
void BatchedGemmIndirect(const size_t m, const size_t n, const size_t k, const tart::buffer_ptr& alphas,
                         const tart::buffer_ptr& a_buffer, const std::vector<int>& a_offsets, const size_t a_ld,
                         const tart::buffer_ptr& b_buffer, const std::vector<int>& b_offsets, const size_t b_ld,
                         const tart::buffer_ptr& betas, const tart::buffer_ptr& c_buffer, const std::vector<int>& c_offsets,
                         const size_t c_ld, const bool a_do_transpose, const bool b_do_transpose,
                         const bool c_do_transpose, const bool a_conjugate, const bool b_conjugate,
                         const size_t a_one, const size_t a_two, const size_t b_one, const size_t b_two,
                         const size_t c_one, const size_t c_two, const size_t batch_count, tart::device_ptr device,
                         const tart::DType& dtype, const Databases& db);

// Direct version of batched GEMM
void BatchedGemmDirect(const size_t m, const size_t n, const size_t k, const tart::buffer_ptr& alphas,
                       const tart::buffer_ptr& a_buffer, const std::vector<int>& a_offsets, const size_t a_ld,
                       const tart::buffer_ptr& b_buffer, const std::vector<int>& b_offsets, const size_t b_ld,
                       const tart::buffer_ptr& betas, const tart::buffer_ptr& c_buffer, const std::vector<int>& c_offsets,
                       const size_t c_ld, const bool a_do_transpose, const bool b_do_transpose,
                       const bool c_do_transpose, const bool a_conjugate, const bool b_conjugate,
                       const size_t batch_count, tart::device_ptr device, const tart::DType& dtype, const Databases& db);

// =================================================================================================
}	// namespace clblast

// CLBLAST_ROUTINES_XGEMMBATCHED_H_
#endif
