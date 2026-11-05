
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xhad routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XHAD_H_
#define CLBLAST_ROUTINES_XHAD_H_

#include <cstddef>
#include <string>

#include "routine.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
class Xhad : public Routine {
 public:
  // Constructor
  Xhad(const tart::DType& dtype, Queue& queue, EventPointer event, const std::string& name = "HAD");

  // Templated-precision implementation of the routine
  void DoHad(const size_t n, const tart::Scalar& alpha, const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
             const tart::buffer_ptr& y_buffer, const size_t y_offset, const size_t y_inc, const tart::Scalar& beta,
             const tart::buffer_ptr& z_buffer, const size_t z_offset, const size_t z_inc);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XHAD_H_
#endif
