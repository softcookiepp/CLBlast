
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xgerc routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XGERC_H_
#define CLBLAST_ROUTINES_XGERC_H_

#include <cstddef>
#include <string>

#include "routines/level2/xger.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
class Xgerc : public Xger {
 public:
  // Uses the regular Xger routine
  using Xger::DoGer;

  // Constructor
  Xgerc(const tart::DType& dtype, Queue& queue, EventPointer event, const std::string& name = "GERC");

  // Templated-precision implementation of the routine
  void DoGerc(const Layout layout, const size_t m, const size_t n, const tart::Scalar& alpha, const tart::buffer_ptr& x_buffer,
              const size_t x_offset, const size_t x_inc, const tart::buffer_ptr& y_buffer, const size_t y_offset,
              const size_t y_inc, const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XGERC_H_
#endif
