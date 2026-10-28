
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xsyr2 routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XSYR2_H_
#define CLBLAST_ROUTINES_XSYR2_H_

#include <cstddef>
#include <string>

#include "routines/level2/xher2.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
class Xsyr2 : public Xher2 {
 public:
  // Uses the regular Xher2 routine
  using Xher2::DoHer2;

  // Constructor
  Xsyr2(const tart::DType& dtype, Queue& queue, EventPointer event, const std::string& name = "SYR2");

  // Templated-precision implementation of the routine
  void DoSyr2(const Layout layout, const Triangle triangle, const size_t n, const tart::Scalar& alpha, const tart::buffer_ptr& x_buffer,
              const size_t x_offset, const size_t x_inc, const tart::buffer_ptr& y_buffer, const size_t y_offset,
              const size_t y_inc, const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XSYR2_H_
#endif
