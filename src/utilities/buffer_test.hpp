
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
void TestMatrixA(const tart::DType& dtype, const size_t one, const size_t two, const tart::buffer_ptr& buffer, const size_t offset, const size_t ld, const bool test_lead_dim = true);

// Tests matrix 'B' for validity
void TestMatrixB(const tart::DType& dtype, const size_t one, const size_t two, const tart::buffer_ptr& buffer, const size_t offset, const size_t ld, const bool test_lead_dim = true);

// Tests matrix 'C' for validity
void TestMatrixC(const tart::DType& dtype, const size_t one, const size_t two, const tart::buffer_ptr& buffer, const size_t offset, const size_t ld, const bool test_lead_dim = true);

// Tests matrix 'AP' for validity
void TestMatrixAP(const tart::DType& dtype, const size_t n, const tart::buffer_ptr& buffer, const size_t offset);

// =================================================================================================

// Tests vector 'X' for validity
void TestVectorX(const tart::DType& dtype, const size_t n, const tart::buffer_ptr& buffer, const size_t offset, const size_t inc);

// Tests vector 'Y' for validity
void TestVectorY(const tart::DType& dtype, const size_t n, const tart::buffer_ptr& buffer, const size_t offset, const size_t inc);

// Tests vector 'Z' for validity
void TestVectorZ(const tart::DType& dtype, const size_t n, const tart::buffer_ptr& buffer, const size_t offset, const size_t inc);

// =================================================================================================

// Tests vector 'scalar' for validity
void TestVectorScalar(const tart::DType& dtype, const size_t n, const tart::buffer_ptr& buffer, const size_t offset);

// Tests vector 'index' for validity
void TestVectorIndex(const tart::DType& dtype, const size_t n, const tart::buffer_ptr& buffer, const size_t offset);

// =================================================================================================

// Tests matrix 'A' for validity in a batched setting
inline void TestBatchedMatrixA(const tart::DType& dtype, const size_t one, const size_t two, const tart::buffer_ptr& buffer, const std::vector<size_t>& offsets,
												const size_t ld, const bool test_lead_dim = true) {
	const auto max_offset = *std::max_element(offsets.begin(), offsets.end());
	TestMatrixA(dtype, one, two, buffer, max_offset, ld, test_lead_dim);
}

// Tests matrix 'B' for validity in a batched setting
inline void TestBatchedMatrixB(const tart::DType& dtype, const size_t one, const size_t two, const tart::buffer_ptr& buffer, const std::vector<size_t>& offsets,
												const size_t ld, const bool test_lead_dim = true) {
	const auto max_offset = *std::max_element(offsets.begin(), offsets.end());
	TestMatrixB(dtype, one, two, buffer, max_offset, ld, test_lead_dim);
}

// Tests matrix 'C' for validity in a batched setting
inline void TestBatchedMatrixC(const tart::DType& dtype, const size_t one, const size_t two, const tart::buffer_ptr& buffer, const std::vector<size_t>& offsets,
												const size_t ld) {
	const auto max_offset = *std::max_element(offsets.begin(), offsets.end());
	TestMatrixC(dtype, one, two, buffer, max_offset, ld);
}

// =================================================================================================

// Tests matrix 'A' for validity in a strided batched setting
inline void TestStridedBatchedMatrixA(const tart::DType& dtype, const size_t one, const size_t two, const tart::buffer_ptr& buffer, const size_t offset,
															 const size_t stride, const size_t batch_count, const size_t ld,
															 const bool test_lead_dim = true) {
	const auto last_batch_offset = (batch_count - 1) * stride;
	TestMatrixA(dtype, one, two, buffer, offset + last_batch_offset, ld, test_lead_dim);
}

// Tests matrix 'B' for validity in a strided batched setting
inline void TestStridedBatchedMatrixB(const tart::DType& dtype, const size_t one, const size_t two, const tart::buffer_ptr& buffer, const size_t offset,
															 const size_t stride, const size_t batch_count, const size_t ld,
															 const bool test_lead_dim = true) {
	const auto last_batch_offset = (batch_count - 1) * stride;
	TestMatrixB(dtype, one, two, buffer, offset + last_batch_offset, ld, test_lead_dim);
}

// Tests matrix 'C' for validity in a strided batched setting
inline void TestStridedBatchedMatrixC(const tart::DType& dtype, const size_t one, const size_t two, const tart::buffer_ptr& buffer, const size_t offset,
															 const size_t stride, const size_t batch_count, const size_t ld) {
	const auto last_batch_offset = (batch_count - 1) * stride;
	TestMatrixC(dtype, one, two, buffer, offset + last_batch_offset, ld);
}

// =================================================================================================
}	// namespace clblast

// CLBLAST_BUFFER_TEST_H_
#endif
