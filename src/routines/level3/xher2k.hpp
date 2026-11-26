
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xher2k routine. The precision is implemented using the template argument
// 'T', whereas the alpha/beta arguments are of type 'U'. The implementation is very similar to the
// Xsyr2k routine.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XHER2K_H_
#define CLBLAST_ROUTINES_XHER2K_H_

#include <cstddef>
#include <string>

#include "routines/level3/xherk.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
class Xher2k {
 public:

   // Constructor
   Xher2k(const tart::DType& dtype, Queue& queue, const std::string& name = "HER2K");

   // Templated-precision implementation of the routine
   void DoHer2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose, const uint32_t n,
                const uint32_t k, const tart::Scalar& alpha, const tart::buffer_ptr& a_buffer, const size_t a_offset, const size_t a_ld,
                const tart::buffer_ptr& b_buffer, const size_t b_offset, const size_t b_ld, const tart::Scalar& beta,
                const tart::buffer_ptr& c_buffer, const size_t c_offset, const size_t c_ld,
                tart::device_ptr device, const tart::DType& dtype, const Databases& db);
};

// =================================================================================================
}  // namespace clblast

// CLBLAST_ROUTINES_XHER2K_H_
#endif
