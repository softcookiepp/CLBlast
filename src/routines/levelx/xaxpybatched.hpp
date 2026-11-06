
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the XaxpyBatched routine. This is a non-blas batched version of AXPY.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XAXPYBATCHED_H_
#define CLBLAST_ROUTINES_XAXPYBATCHED_H_

#include <cstddef>
#include <string>
#include <vector>

#include "routine.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
class XaxpyBatched : public Routine
{
 public:
	// Constructor
	XaxpyBatched(const tart::DType& dtype, Queue& queue, EventPointer event, const std::string& name = "AXPYBATCHED");

	// Templated-precision implementation of the routine
	void DoAxpyBatched(const size_t n, const tart::buffer_ptr& alphas, const tart::buffer_ptr& x_buffer,
		const std::vector<size_t>& x_offsets, const size_t x_inc, const tart::buffer_ptr& y_buffer,
		const std::vector<size_t>& y_offsets, const size_t y_inc, const size_t batch_count);
	
	template <typename T>
	void DoAxpyBatched(const size_t n, const std::vector<T>& alphas, const tart::buffer_ptr& x_buffer,
		const std::vector<size_t>& x_offsets, const size_t x_inc, const tart::buffer_ptr& y_buffer,
		const std::vector<size_t>& y_offsets, const size_t y_inc, const size_t batch_count)
	{
		// Tests for a valid batch count
		if ((batch_count < 1) || (alphas.size() != batch_count) || (x_offsets.size() != batch_count) ||
				(y_offsets.size() != batch_count)) {
			throw BLASError(StatusCode::kInvalidBatchCount);
		}
		if (alphas.size() == 0)
			std::cout << "\nalphas.size(): " << alphas.size() << std::endl;
		tart::buffer_ptr alphas_device = this->mDevice->allocateBuffer(alphas.size()*sizeof(T));
		alphas_device->copyIn(alphas);
		DoAxpyBatched(n, alphas_device, x_buffer, x_offsets, x_inc, y_buffer, y_offsets, y_inc, batch_count);
	}
};

// =================================================================================================
}	// namespace clblast

// CLBLAST_ROUTINES_XAXPYBATCHED_H_
#endif
