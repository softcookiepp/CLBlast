
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the functional interface for the Xgemv routine (matrix-vector multiplication).
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XGEMV_H_
#define CLBLAST_ROUTINES_XGEMV_H_

#include <cstddef>
#include <string>

#include "database/database.hpp"
#include "utilities/backend.hpp"
#include "tart/dtypes.hpp"

namespace clblast {
// =================================================================================================

// Functional implementation of Xgemv (matrix-vector multiplication)
void DoGemv(const Layout layout, const Transpose a_transpose, const size_t m, const size_t n,
            const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld,
            const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
            const tart::Scalar& beta, const tart::buffer_ptr& y_buffer, const size_t y_offset, const size_t y_inc,
            const Databases& db);

// Generic version used also for other matrix-vector multiplications
void MatVec(const Layout layout, const Transpose a_transpose, const size_t m, const size_t n,
            const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld,
            const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
            const tart::Scalar& beta, const tart::buffer_ptr& y_buffer, const size_t y_offset, const size_t y_inc,
            bool fast_kernel, bool fast_kernel_rot, const size_t parameter, const bool packed,
            const size_t kl, const size_t ku, const std::string& name, const Databases& db,
            tart::device_ptr device, const tart::DType& dtype);

// Minimal base class for Xgemv-derived routines
class Xgemv {
 public:
	// Constructor
	Xgemv(const tart::DType& dtype, Queue& queue, EventPointer event, const std::string& name = "GEMV");

	// Templated-precision implementation of the routine
	void DoGemv(const Layout layout, const Transpose a_transpose, const size_t m, const size_t n,
							const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld,
							const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
							const tart::Scalar& beta, const tart::buffer_ptr& y_buffer, const size_t y_offset, const size_t y_inc,
							const Databases& db,
							tart::device_ptr device = nullptr, const tart::DType& dtype = tart::dtypes::unknown_type);

	// Generic version used also for other matrix-vector multiplications
void MatVec(const Layout layout, const Transpose a_transpose, const size_t m, const size_t n,
						const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld,
						const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
						const tart::Scalar& beta, const tart::buffer_ptr& y_buffer, const size_t y_offset, const size_t y_inc,
						bool fast_kernel, bool fast_kernel_rot, const size_t parameter, const bool packed,
						const size_t kl, const size_t ku, const std::string& name, const Databases& db,
						tart::device_ptr device = nullptr, const tart::DType& dtype = tart::dtypes::unknown_type);
};

// =================================================================================================
}	// namespace clblast

// CLBLAST_ROUTINES_XGEMV_H_
#endif
