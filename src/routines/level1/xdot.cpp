
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the functional Xdot routine.
//
// =================================================================================================

#include "routines/level1/xdot.hpp"
#include "routine_aot_cache.hpp"
#include <cstddef>
#include <string>
#include <vector>

#include "routines/common.hpp"
#include "utilities/backend.hpp"
#include "utilities/buffer_test.hpp"
#include "utilities/clblast_exceptions.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// Functional implementation of Xdot
void DoDot(const tart::DType& dtype, const tart::device_ptr& device,
           const size_t n, const tart::buffer_ptr& dot_buffer, const size_t dot_offset,
           const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
           const tart::buffer_ptr& y_buffer, const size_t y_offset, const size_t y_inc,
           const bool do_conjugate, const Databases& db)
{
	// Makes sure all dimensions are larger than zero
	if (n == 0) {
		throw BLASError(StatusCode::kInvalidDimension);
	}

	// Tests the vectors for validity
	TestVectorX(dtype, n, x_buffer, x_offset, x_inc);
	TestVectorY(dtype, n, y_buffer, y_offset, y_inc);
	TestVectorScalar(dtype, 1, dot_buffer, dot_offset);

	// Retrieves the Xdot kernels from the compiled binary
	tart::kernel_ptr kernel1 = getDot(device, dtype);
	tart::kernel_ptr kernel2 = getDotEpilogue(device, dtype);

	// Creates the buffer for intermediate values
	auto temp_size = 2 * db["WGS2"] * dtype.size();
	auto temp_buffer = device->allocateBuffer(temp_size);
	
	// Sets the kernel arguments
#if VULKAN_USE_BDA
	tart::DeviceMetadata meta = device->getMetadata();
	if (meta.bda)
	{
		kernel1->setArg(0, static_cast<int>(n));
		kernel1->setArg(1, x_buffer->getAddress());
		kernel1->setArg(2, static_cast<int>(x_offset));
		kernel1->setArg(3, static_cast<int>(x_inc));
		kernel1->setArg(4, y_buffer->getAddress());
		kernel1->setArg(5, static_cast<int>(y_offset));
		kernel1->setArg(6, static_cast<int>(y_inc));
		kernel1->setArg(7, temp_buffer->getAddress());
		kernel1->setArg(8, static_cast<int>(do_conjugate));
	}
	else
#endif
	{
		kernel1->setArg(0, static_cast<int>(n));
		kernel1->setArg(1, x_buffer);
		kernel1->setArg(2, static_cast<int>(x_offset));
		kernel1->setArg(3, static_cast<int>(x_inc));
		kernel1->setArg(4, y_buffer);
		kernel1->setArg(5, static_cast<int>(y_offset));
		kernel1->setArg(6, static_cast<int>(y_inc));
		kernel1->setArg(7, temp_buffer);
		kernel1->setArg(8, static_cast<int>(do_conjugate));
	}

	// Launches the main kernel
	auto global1 = std::vector<size_t>{db["WGS1"] * temp_size};
	auto local1 = std::vector<size_t>{db["WGS1"]};
	kernel1->setArg(9, static_cast<int>(temp_size));
	kernel1->enqueue({temp_size}, {db["WGS1"], 1});
	
	// Sets the arguments for the epilogue kernel
	#if VULKAN_USE_BDA
	if (meta.bda)
	{
		kernel2->setArg(0, temp_buffer->getAddress());
		kernel2->setArg(1, dot_buffer->getAddress());
		kernel2->setArg(2, static_cast<int>(dot_offset));
	}
	else
	#endif
	{
		kernel2->setArg(0, temp_buffer);
		kernel2->setArg(1, dot_buffer);
		kernel2->setArg(2, static_cast<int>(dot_offset));
	}

	// Launches the epilogue kernel
	auto global2 = std::vector<size_t>{db["WGS2"]};
	auto local2 = std::vector<size_t>{db["WGS2"]};
	kernel2->enqueue({1}, {db["WGS2"], 1});
}

// =================================================================================================

}	// namespace clblast
