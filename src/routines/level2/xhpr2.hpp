
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xhpr2 routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XHPR2_H_
#define CLBLAST_ROUTINES_XHPR2_H_

#include <cstddef>
#include <string>

#include "routines/level2/xher2.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
class Xhpr2 : public Xher2 {
 public:
  // Uses the regular Xher2 routine
  using Xher2::DoHer2;

  // Constructor
  Xhpr2(const tart::DType& dtype, Queue& queue, EventPointer event, const std::string& name = "HPR2");

  // Templated-precision implementation of the routine
  void DoHpr2(const Layout layout, const Triangle triangle, const size_t n, const tart::Scalar& alpha, const tart::buffer_ptr& x_buffer,
              const size_t x_offset, const size_t x_inc, const tart::buffer_ptr& y_buffer, const size_t y_offset,
              const size_t y_inc, const tart::buffer_ptr& ap_buffer, const size_t ap_offset);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XHPR2_H_
#endif
