
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xaxpy routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XAXPY_H_
#define CLBLAST_ROUTINES_XAXPY_H_

#include <cstddef>
#include <string>

#include "routine.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xaxpy : public Routine {
 public:
  // Constructor
  Xaxpy(Queue& queue, EventPointer event, const std::string& name = "AXPY");

  // Templated-precision implementation of the routine
  void DoAxpy(const size_t n, const tart::Scalar& alpha, const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
              const tart::buffer_ptr& y_buffer, const size_t y_offset, const size_t y_inc);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XAXPY_H_
#endif
