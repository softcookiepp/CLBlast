
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the XaxpyBatched class (see the header for information about the class).
//
// =================================================================================================

#include "routines/levelx/xaxpybatched.hpp"

#include <cstddef>
#include <string>
#include <vector>

#include "routine.hpp"
#include "routines/common.hpp"
#include "utilities/backend.hpp"
#include "utilities/buffer_test.hpp"
#include "utilities/clblast_exceptions.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
XaxpyBatched::XaxpyBatched(const tart::DType& dtype, Queue& queue, EventPointer event, const std::string& name)
		: Routine(queue, event, name, {"Xaxpy"}, dtype, {},
							{
	#include "../../kernels-vk-inline/level1/xaxpy_batched.glsl.inl"
							}
,
 {"XaxpyBatched"}
			) {
}

// =================================================================================================

// The main routine
void XaxpyBatched::DoAxpyBatched(const size_t n, const tart::buffer_ptr& alphas_device, const tart::buffer_ptr& x_buffer,
	const std::vector<size_t>& x_offsets, const size_t x_inc, const tart::buffer_ptr& y_buffer,
	const std::vector<size_t>& y_offsets, const size_t y_inc,
	const size_t batch_count)
{
	// Makes sure all dimensions are larger than zero
	if (n == 0) {
		throw BLASError(StatusCode::kInvalidDimension);
	}

	// Tests the vectors for validity
	for (auto batch = size_t{0}; batch < batch_count; ++batch) {
		TestVectorX(mDtype, n, x_buffer, x_offsets[batch], x_inc);
		TestVectorY(mDtype, n, y_buffer, y_offsets[batch], y_inc);
	}

	// Upload the arguments to the device
	auto x_offsets_int = std::vector<int>(batch_count);
	auto y_offsets_int = std::vector<int>(batch_count);
	for (auto batch = size_t{0}; batch < batch_count; ++batch)
	{
		x_offsets_int[batch] = static_cast<int>(x_offsets[batch]);
		y_offsets_int[batch] = static_cast<int>(y_offsets[batch]);
	}
	auto x_offsets_device = this->mDevice->allocateBuffer(x_offsets_int.size()*sizeof(int));
	x_offsets_device->copyIn(x_offsets_int);
	auto y_offsets_device = this->mDevice->allocateBuffer(y_offsets_int.size()*sizeof(int));
	y_offsets_device->copyIn(y_offsets_int);

	// Retrieves the Xaxpy kernel from the compiled binary
	tart::kernel_ptr kernel = getAxpyBatched(mDevice, mDtype);

	// Sets the kernel arguments
	kernel->setArg(0, static_cast<int>(n));
	kernel->setArg(1, alphas_device);
	kernel->setArg(2, x_buffer);
	kernel->setArg(3, x_offsets_device);
	kernel->setArg(4, static_cast<int>(x_inc));
	kernel->setArg(5, y_buffer);
	kernel->setArg(6, y_offsets_device);
	kernel->setArg(7, static_cast<int>(y_inc));

	// Launches the kernel
	auto n_ceiled = Ceil(n, db_["WGS"] * db_["WPT"]);
	auto global = std::vector<size_t>{n_ceiled / db_["WPT"], batch_count};
	auto local = std::vector<size_t>{db_["WGS"], 1};
	kernel->enqueue({global[0], global[1], 1}, {db_["WGS"], db_["WPT"]});
}

// =================================================================================================
}	// namespace clblast
