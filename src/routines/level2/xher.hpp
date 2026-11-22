
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the functional interface for the Xher routine (Hermitian rank-1 matrix update).
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XHER_H_
#define CLBLAST_ROUTINES_XHER_H_

#include <cstddef>
#include <string>

#include "database/database.hpp"
#include "utilities/backend.hpp"
#include "tart/dtypes.hpp"

namespace clblast {
// =================================================================================================

// Functional implementation of Xher (Hermitian rank-1 matrix update)
void DoHer(const Layout layout, const Triangle triangle, const size_t n, const tart::Scalar& alpha,
           const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
           const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld,
           const bool packed, const std::string& name, const Databases& db,
           tart::device_ptr device, const tart::DType& dtype);

// Minimal base class for Xher-derived routines
class Xher {
 public:
	// Constructor
	Xher(const tart::DType& dtype, Queue& queue, EventPointer event, const std::string& name = "HER");

	// Templated-precision implementation of the routine
	void DoHer(const Layout layout, const Triangle triangle, const size_t n, const tart::Scalar& alpha,
							const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
							const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld,
							const bool packed = false, const std::string& name = "HER",
							const Databases& db = Databases({"Her"}),
							tart::device_ptr device = nullptr, const tart::DType& dtype = tart::dtypes::unknown_type);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XHER_H_
#endif
