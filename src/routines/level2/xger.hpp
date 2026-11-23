
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the functional interface for the Xger routine (general rank-1 matrix update).
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XGER_H_
#define CLBLAST_ROUTINES_XGER_H_

#include <cstddef>
#include <string>

#include "database/database.hpp"
#include "utilities/backend.hpp"
#include "tart/dtypes.hpp"

namespace clblast {
// =================================================================================================

// Functional implementation of Xger (general rank-1 matrix update)
void DoGer(const Layout layout, const size_t m, const size_t n, const tart::Scalar& alpha,
           const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
           const tart::buffer_ptr& y_buffer, const size_t y_offset, const size_t y_inc,
           const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld,
           const std::string& name, const Databases& db);

// Minimal base class for Xger-derived routines
class Xger {
 public:
	// Constructor
	Xger(const tart::DType& dtype, Queue& queue, const std::string& name = "GER");

	// Templated-precision implementation of the routine
	void DoGer(const Layout layout, const size_t m, const size_t n, const tart::Scalar& alpha,
							const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
							const tart::buffer_ptr& y_buffer, const size_t y_offset, const size_t y_inc,
							const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld,
							const std::string& name = "GER", const Databases& db = Databases({"Ger"}),
							tart::device_ptr device = nullptr, const tart::DType& dtype = tart::dtypes::unknown_type);
};

// =================================================================================================
}	// namespace clblast

// CLBLAST_ROUTINES_XGER_H_
#endif
