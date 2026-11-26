
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the tests for the OpenCL buffers (matrices and vectors). These tests are
// templated and thus header-only.
//
// =================================================================================================

#ifndef CLBLAST_BUFFER_TEST_H_
#define CLBLAST_BUFFER_TEST_H_

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "utilities/backend.hpp"
#include "utilities/clblast_exceptions.hpp"

namespace clblast {
// =================================================================================================

// Tests matrix 'A' for validity
void TestMatrixA([[maybe_unused]] const tart::DType& dtype, const uint32_t one, const uint32_t two, const tart::buffer_ptr& buffer, const uint32_t offset, const uint32_t ld, const bool test_lead_dim = true);

// Tests matrix 'B' for validity
void TestMatrixB([[maybe_unused]] const tart::DType& dtype, const uint32_t one, const uint32_t two, const tart::buffer_ptr& buffer, const uint32_t offset, const uint32_t ld, const bool test_lead_dim = true);

// Tests matrix 'C' for validity
void TestMatrixC([[maybe_unused]] const tart::DType& dtype, const uint32_t one, const uint32_t two, const tart::buffer_ptr& buffer, const uint32_t offset, const uint32_t ld, const bool test_lead_dim = true);

// Tests matrix 'AP' for validity
void TestMatrixAP([[maybe_unused]] const tart::DType& dtype, const uint32_t n, const tart::buffer_ptr& buffer, const uint32_t offset);

// =================================================================================================

// Tests vector 'X' for validity
void TestVectorX([[maybe_unused]] const tart::DType& dtype, const uint32_t n, const tart::buffer_ptr& buffer, const uint32_t offset, const uint32_t inc);

// Tests vector 'Y' for validity
void TestVectorY([[maybe_unused]] const tart::DType& dtype, const uint32_t n, const tart::buffer_ptr& buffer, const uint32_t offset, const uint32_t inc);

// Tests vector 'Z' for validity
void TestVectorZ([[maybe_unused]] const tart::DType& dtype, const uint32_t n, const tart::buffer_ptr& buffer, const uint32_t offset, const uint32_t inc);

// =================================================================================================

// Tests vector 'scalar' for validity
void TestVectorScalar([[maybe_unused]] const tart::DType& dtype, const uint32_t n, const tart::buffer_ptr& buffer, const uint32_t offset);

// Tests vector 'index' for validity
void TestVectorIndex([[maybe_unused]] const tart::DType& dtype, const uint32_t n, const tart::buffer_ptr& buffer, const uint32_t offset);

// =================================================================================================

// Tests matrix 'A' for validity in a batched setting
inline void TestBatchedMatrixA([[maybe_unused]] const tart::DType& dtype, const uint32_t one, const uint32_t two, const tart::buffer_ptr& buffer, const std::vector<uint32_t>& offsets,
												const uint32_t ld, const bool test_lead_dim = true) {
	const auto max_offset = *std::max_element(offsets.begin(), offsets.end());
	TestMatrixA(dtype, one, two, buffer, max_offset, ld, test_lead_dim);
}

// Tests matrix 'B' for validity in a batched setting
inline void TestBatchedMatrixB([[maybe_unused]] const tart::DType& dtype, const uint32_t one, const uint32_t two, const tart::buffer_ptr& buffer, const std::vector<uint32_t>& offsets,
												const uint32_t ld, const bool test_lead_dim = true) {
	const auto max_offset = *std::max_element(offsets.begin(), offsets.end());
	TestMatrixB(dtype, one, two, buffer, max_offset, ld, test_lead_dim);
}

// Tests matrix 'C' for validity in a batched setting
inline void TestBatchedMatrixC([[maybe_unused]] const tart::DType& dtype, const uint32_t one, const uint32_t two, const tart::buffer_ptr& buffer, const std::vector<uint32_t>& offsets,
												const uint32_t ld) {
	const auto max_offset = *std::max_element(offsets.begin(), offsets.end());
	TestMatrixC(dtype, one, two, buffer, max_offset, ld);
}

// =================================================================================================

// Tests matrix 'A' for validity in a strided batched setting
inline void TestStridedBatchedMatrixA([[maybe_unused]] const tart::DType& dtype, const uint32_t one, const uint32_t two, const tart::buffer_ptr& buffer, const uint32_t offset,
															 const uint32_t stride, const uint32_t batch_count, const uint32_t ld,
															 const bool test_lead_dim = true) {
	const auto last_batch_offset = (batch_count - 1) * stride;
	TestMatrixA(dtype, one, two, buffer, offset + last_batch_offset, ld, test_lead_dim);
}

// Tests matrix 'B' for validity in a strided batched setting
inline void TestStridedBatchedMatrixB([[maybe_unused]] const tart::DType& dtype, const uint32_t one, const uint32_t two, const tart::buffer_ptr& buffer, const uint32_t offset,
															 const uint32_t stride, const uint32_t batch_count, const uint32_t ld,
															 const bool test_lead_dim = true) {
	const auto last_batch_offset = (batch_count - 1) * stride;
	TestMatrixB(dtype, one, two, buffer, offset + last_batch_offset, ld, test_lead_dim);
}

// Tests matrix 'C' for validity in a strided batched setting
inline void TestStridedBatchedMatrixC([[maybe_unused]] const tart::DType& dtype, const uint32_t one, const uint32_t two, const tart::buffer_ptr& buffer, const uint32_t offset,
															 const uint32_t stride, const uint32_t batch_count, const uint32_t ld) {
	const auto last_batch_offset = (batch_count - 1) * stride;
	TestMatrixC(dtype, one, two, buffer, offset + last_batch_offset, ld);
}

// =================================================================================================
}	// namespace clblast

// CLBLAST_BUFFER_TEST_H_
#endif
