
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xgerc class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level2/xgerc.hpp"

#include <cstddef>
#include <string>

#include "routines/level2/xger.hpp"
#include "utilities/backend.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
Xgerc::Xgerc(const tart::DType& dtype, Queue& queue, EventPointer event, const std::string& name) : Xger(dtype, queue, event, name) {}

// =================================================================================================

// The main routine
void Xgerc::DoGerc(const Layout layout, const size_t m, const size_t n, const tart::Scalar& alpha, const tart::buffer_ptr& x_buffer,
                      const size_t x_offset, const size_t x_inc, const tart::buffer_ptr& y_buffer, const size_t y_offset,
                      const size_t y_inc, const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld) {
  // Regular Ger operation on complex data, plus conjugation in the kernel guarded by the
  // ROUTINE_GERC guard.
  DoGer(layout, m, n, alpha, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, a_buffer, a_offset, a_ld);
}

// =================================================================================================
}  // namespace clblast
