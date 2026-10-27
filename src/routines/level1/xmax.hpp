
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xmax routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XMAX_H_
#define CLBLAST_ROUTINES_XMAX_H_

#include <cstddef>
#include <string>

#include "routines/level1/xamax.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
class Xmax : public Xamax {
 public:
  // Members and methods from the base class
  using Xamax::DoAmax;

  // Constructor
  Xmax(const tart::DType& dtype, Queue& queue, EventPointer event, const std::string& name = "MAX") : Xamax(dtype, queue, event, name) {}

  // Forwards to the regular absolute version. The implementation difference is realised in the
  // kernel through a pre-processor macro based on the name of the routine.
  void DoMax(const size_t n, const tart::buffer_ptr& imax_buffer, const size_t imax_offset,
             const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc) {
    DoAmax(n, imax_buffer, imax_offset, x_buffer, x_offset, x_inc);
  }
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XMAX_H_
#endif
