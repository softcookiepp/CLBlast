
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the functional interface for the XaxpyBatched routine.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XAXPYBATCHED_H_
#define CLBLAST_ROUTINES_XAXPYBATCHED_H_

#include <cstddef>
#include <string>
#include <vector>

#include "database/database.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// Functional implementation of XaxpyBatched (batched AXPY)
void DoAxpyBatched(const uint32_t n, const tart::buffer_ptr& alphas_device, const tart::buffer_ptr& x_buffer,
                   const std::vector<uint32_t>& x_offsets, const size_t x_inc, const tart::buffer_ptr& y_buffer,
                   const std::vector<uint32_t>& y_offsets, const size_t y_inc, const uint32_t batch_count,
                   tart::device_ptr device, const tart::DType& dtype, const Databases& db);

// Helper function for template method
template <typename T>
void UploadAxpyBatched(const uint32_t n, const std::vector<T>& alphas, const tart::buffer_ptr& x_buffer,
                       const std::vector<uint32_t>& x_offsets, const size_t x_inc, const tart::buffer_ptr& y_buffer,
                       const std::vector<uint32_t>& y_offsets, const size_t y_inc, const uint32_t batch_count,
                       tart::device_ptr device, const tart::DType& dtype, const Databases& db);

// =================================================================================================
}	// namespace clblast

// CLBLAST_ROUTINES_XAXPYBATCHED_H_
#endif
