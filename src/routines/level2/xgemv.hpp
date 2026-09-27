
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xgemv routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XGEMV_H_
#define CLBLAST_ROUTINES_XGEMV_H_

#include <cstddef>
#include <string>

#include "routine.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xgemv : public Routine
{
	uint32_t mGBMV = 0;
	uint32_t mHEMV = 0;
	uint32_t mSYMV = 0;
	uint32_t mTRMV = 0;
	uint32_t mHBMV = 0;
	uint32_t mSBMV = 0;
	uint32_t mTBMV = 0;
	uint32_t mHPMV = 0;
	uint32_t mSPMV = 0;
	uint32_t mTPMV = 0;
 public:
	// Constructor
	Xgemv(Queue& queue, EventPointer event, const std::string& name = "GEMV");

	// Templated-precision implementation of the routine
	void DoGemv(const Layout layout, const Transpose a_transpose, const size_t m, const size_t n, const T alpha,
							const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld, const Buffer<T>& x_buffer,
							const size_t x_offset, const size_t x_inc, const T beta, const Buffer<T>& y_buffer, const size_t y_offset,
							const size_t y_inc);

	// Generic version used also for other matrix-vector multiplications
	void MatVec(const Layout layout, const Transpose a_transpose, const size_t m, const size_t n, const T alpha,
							const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld, const Buffer<T>& x_buffer,
							const size_t x_offset, const size_t x_inc, const T beta, const Buffer<T>& y_buffer, const size_t y_offset,
							const size_t y_inc, bool fast_kernel, bool fast_kernel_rot, const size_t parameter, const bool packed,
							const size_t kl, const size_t ku);
};

// =================================================================================================
}	// namespace clblast

// CLBLAST_ROUTINES_XGEMV_H_
#endif
