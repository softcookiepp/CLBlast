
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the functional interface for the Xher2 routine (Hermitian rank-2 matrix update).
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XHER2_H_
#define CLBLAST_ROUTINES_XHER2_H_

#include <cstddef>
#include <string>

#include "database/database.hpp"
#include "utilities/backend.hpp"
#include "tart/dtypes.hpp"

namespace clblast {
// =================================================================================================

// Functional implementation of Xher2 (Hermitian rank-2 matrix update)
void DoHer2(const Layout layout, const Triangle triangle, const size_t n, const tart::Scalar& alpha,
            const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
            const tart::buffer_ptr& y_buffer, const size_t y_offset, const size_t y_inc,
            const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld,
            const bool packed, const std::string& name, const Databases& db,
            tart::device_ptr device, const tart::DType& dtype);

// Minimal base class for Xher2-derived routines
class Xher2 {
 public:
	// Constructor
	Xher2(const tart::DType& dtype, Queue& queue, EventPointer event, const std::string& name = "HER2");

	// Templated-precision implementation of the routine
	void DoHer2(const Layout layout, const Triangle triangle, const size_t n, const tart::Scalar& alpha,
							const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
							const tart::buffer_ptr& y_buffer, const size_t y_offset, const size_t y_inc,
							const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld,
							const bool packed = false, const std::string& name = "HER2", const Databases& db = Databases({"Her2"}),
							tart::device_ptr device = nullptr, const tart::DType& dtype = tart::dtypes::unknown_type);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XHER2_H_
#endif
