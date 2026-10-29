
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xhpr class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level2/xhpr.hpp"

#include <cstddef>
#include <string>

#include "routines/level2/xher.hpp"
#include "utilities/backend.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
Xhpr::Xhpr(const tart::DType& dtype, Queue& queue, EventPointer event, const std::string& name) : Xher(dtype, queue, event, name) {}

// =================================================================================================

// The main routine
void Xhpr::DoHpr(const Layout layout, const Triangle triangle, const size_t n, const tart::Scalar& alpha,
                       const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc, const tart::buffer_ptr& ap_buffer,
                       const size_t ap_offset) {
  // Specific Xhpr functionality is implemented in the kernel using defines
  DoHer(layout, triangle, n, alpha, x_buffer, x_offset, x_inc, ap_buffer, ap_offset, n,
        true);  // packed matrix
}

// =================================================================================================
}  // namespace clblast
