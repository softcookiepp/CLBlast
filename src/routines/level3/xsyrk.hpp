
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the functional interface for the Xsyrk routine (Symmetric rank-K update).
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XSYRK_H_
#define CLBLAST_ROUTINES_XSYRK_H_

#include <cstddef>
#include <string>

#include "database/database.hpp"
#include "utilities/backend.hpp"
#include "tart/dtypes.hpp"

namespace clblast {
// =================================================================================================

// Functional implementation of Xsyrk (Symmetric rank-K update)
void DoSyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose, const size_t n, const size_t k,
            const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld,
            const tart::Scalar& beta, const tart::buffer_ptr& c_buffer, const size_t c_offset, const size_t c_ld,
            const Databases& db);

// Helper function to be reused for SYR2K
void SyrkAB(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Transpose b_transpose,
            const size_t n, const size_t k, const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer,
            const size_t a_offset, const size_t a_ld, const tart::buffer_ptr& b_buffer, const size_t b_offset,
            const size_t b_ld, const tart::Scalar& beta, const tart::buffer_ptr& c_buffer, const size_t c_offset,
            const size_t c_ld, const Databases& db,
            tart::device_ptr device, const tart::DType& dtype);

// Minimal base class for Xsyrk-derived routines
class Xsyrk {
 public:
	// Constructor
	Xsyrk(const tart::DType& dtype, Queue& queue, const std::string& name = "SYRK");

	// Templated-precision implementation of the routine
	void DoSyrk(const Layout layout, const Triangle triangle, const Transpose a_transpose, const size_t n, const size_t k,
							const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld,
							const tart::Scalar& beta, const tart::buffer_ptr& c_buffer, const size_t c_offset, const size_t c_ld,
							const Databases& db = Databases({"Syrk"}),
							tart::device_ptr device = nullptr, const tart::DType& dtype = tart::dtypes::unknown_type);

	// Helper function to be reused for SYR2K
	void SyrkAB(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Transpose b_transpose,
							const size_t n, const size_t k, const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer,
							const size_t a_offset, const size_t a_ld, const tart::buffer_ptr& b_buffer, const size_t b_offset,
							const size_t b_ld, const tart::Scalar& beta, const tart::buffer_ptr& c_buffer, const size_t c_offset,
							const size_t c_ld, const Databases& db,
							tart::device_ptr device = nullptr, const tart::DType& dtype = tart::dtypes::unknown_type);
};

// =================================================================================================
}	// namespace clblast

// CLBLAST_ROUTINES_XSYRK_H_
#endif
