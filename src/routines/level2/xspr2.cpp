
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xspr2 class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level2/xspr2.hpp"

#include <cstddef>
#include <string>

#include "routines/level2/xher2.hpp"
#include "utilities/backend.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
Xspr2::Xspr2(const tart::DType& dtype, Queue& queue, EventPointer event, const std::string& name) : Xher2(dtype, queue, event, name) {}

// =================================================================================================

// The main routine
void Xspr2::DoSpr2(const Layout layout, const Triangle triangle, const size_t n, const tart::Scalar& alpha,
                      const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc, const tart::buffer_ptr& y_buffer,
                      const size_t y_offset, const size_t y_inc, const tart::buffer_ptr& ap_buffer, const size_t ap_offset) {
  // Specific Xspr2 functionality is implemented in the kernel using defines
  DoHer2(layout, triangle, n, alpha, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, ap_buffer, ap_offset, n,
         true);  // packed matrix
}

// =================================================================================================
}  // namespace clblast
