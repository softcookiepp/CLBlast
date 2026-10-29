
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xhpr routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XHPR_H_
#define CLBLAST_ROUTINES_XHPR_H_

#include <cstddef>
#include <string>

#include "routines/level2/xher.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
class Xhpr : public Xher {
 public:
  // Uses the regular Xher routine
  using Xher::DoHer;

  // Constructor
  Xhpr(const tart::DType& dtype, Queue& queue, EventPointer event, const std::string& name = "HPR");

  // Templated-precision implementation of the routine
  void DoHpr(const Layout layout, const Triangle triangle, const size_t n, const tart::Scalar& alpha, const tart::buffer_ptr& x_buffer,
             const size_t x_offset, const size_t x_inc, const tart::buffer_ptr& ap_buffer, const size_t ap_offset);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XHPR_H_
#endif
