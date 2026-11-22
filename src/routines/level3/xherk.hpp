
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the functional interface for the Xherk routine (Hermitian rank-K update).
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XHERK_H_
#define CLBLAST_ROUTINES_XHERK_H_

#include <cstddef>
#include <string>

#include "database/database.hpp"
#include "utilities/backend.hpp"
#include "tart/dtypes.hpp"

namespace clblast {
// =================================================================================================

// Functional implementation of Xherk (Hermitian rank-K update)
void DoHerk(const Layout layout, const Triangle triangle, const Transpose a_transpose, const size_t n, const size_t k,
            const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld,
            const tart::Scalar& beta, const tart::buffer_ptr& c_buffer, const size_t c_offset, const size_t c_ld,
            const Databases& db);

// Helper function to be reused for HER2K
void HerkAB(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Transpose b_transpose,
            const size_t n, const size_t k, const tart::Scalar& complex_alpha, const tart::buffer_ptr& a_buffer,
            const size_t a_offset, const size_t a_ld, const tart::buffer_ptr& b_buffer, const size_t b_offset,
            const size_t b_ld, const tart::Scalar& complex_beta, const tart::buffer_ptr& c_buffer,
            const size_t c_offset, const size_t c_ld, const bool diagonal_to_zero, const Databases& db,
            tart::device_ptr device, const tart::DType& dtype);

// Minimal base class for Xherk-derived routines
class Xherk {
 public:
	// Constructor
	Xherk(const tart::DType& dtype, Queue& queue, EventPointer event, const std::string& name = "HERK");

	// Templated-precision implementation of the routine
	void DoHerk(const Layout layout, const Triangle triangle, const Transpose a_transpose, const size_t n, const size_t k,
							const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld,
							const tart::Scalar& beta, const tart::buffer_ptr& c_buffer, const size_t c_offset, const size_t c_ld,
							const Databases& db = Databases({"Herk"}),
							tart::device_ptr device = nullptr, const tart::DType& dtype = tart::dtypes::unknown_type);

	// Helper function to be reused for HER2K
	void HerkAB(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Transpose b_transpose,
							const size_t n, const size_t k, const tart::Scalar& complex_alpha, const tart::buffer_ptr& a_buffer,
							const size_t a_offset, const size_t a_ld, const tart::buffer_ptr& b_buffer, const size_t b_offset,
							const size_t b_ld, const tart::Scalar& complex_beta, const tart::buffer_ptr& c_buffer,
							const size_t c_offset, const size_t c_ld, const bool diagonal_to_zero = false,
							const Databases& db = Databases({"Herk"}),
							tart::device_ptr device = nullptr, const tart::DType& dtype = tart::dtypes::unknown_type);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XHERK_H_
#endif
