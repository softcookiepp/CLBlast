
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the functional Xamax routine.
//
// =================================================================================================

#include "routines/level1/xamax.hpp"
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

// Functional implementation of Xamax (index of absolute maximum)
void DoAmax(const tart::DType& dtype, const tart::device_ptr& device,
            const uint32_t n, const tart::buffer_ptr& imax_buffer, const size_t imax_offset,
            const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
            uint32_t max_flag, uint32_t min_flag, uint32_t amin_flag, const Databases& db) {
	// Makes sure all dimensions are larger than zero
	if (n == 0) {
		throw BLASError(StatusCode::kInvalidDimension);
	}

	// Tests the vectors for validity
	TestVectorX(dtype, n, x_buffer, x_offset, x_inc);
	TestVectorIndex(dtype, 1, imax_buffer, imax_offset);

	// Retrieves the Xamax kernels from the compiled binary
	tart::kernel_ptr kernel1 = getAmax(device, dtype);
	tart::kernel_ptr kernel2 = getAmaxEpilogue(device, dtype);

	// Creates the buffer for intermediate values
	auto temp_size = 2 * db["WGS2"];
	auto temp_buffer1 = device->allocateBuffer(temp_size*dtype.size());
	auto temp_buffer2 = device->allocateBuffer(temp_size*sizeof(uint32_t));

	// Sets the kernel arguments
#if VULKAN_USE_BDA
	const tart::DeviceMetadata meta = device->getMetadata();
	if (meta.bda)
	{
		kernel1->setArg(0, static_cast<int>(n));
		kernel1->setArg(1, x_buffer->getAddress() + x_offset*dtype.size());
		kernel1->setArg(2, static_cast<int>(0));
		kernel1->setArg(3, static_cast<int>(x_inc));
		kernel1->setArg(4, temp_buffer1->getAddress());
		kernel1->setArg(5, temp_buffer2->getAddress());
	}
	else
#endif
	{
		kernel1->setArg(0, static_cast<int>(n));
		kernel1->setArg(1, x_buffer);
		kernel1->setArg(2, static_cast<int>(x_offset));
		kernel1->setArg(3, static_cast<int>(x_inc));
		kernel1->setArg(4, temp_buffer1);
		kernel1->setArg(5, temp_buffer2);
	}

	// Launches the main kernel
	//auto global1 = std::vector<size_t>{db["WGS1"] * temp_size};
	auto global1 = std::vector<size_t>{static_cast<uint32_t>(temp_size)};
	//auto local1 = std::vector<uint32_t>{db["WGS1"]};
	
	// the number of workgroups in the X dimension
	int num_groups_0 = static_cast<int>(global1[0]);
	kernel1->setArg(6, num_groups_0);
	
	kernel1->enqueue(global1, {static_cast<uint32_t>(db["WGS1"]), max_flag, min_flag, amin_flag});

	// Sets the arguments for the epilogue kernel
#if VULKAN_USE_BDA
	kernel2->setArg(0, temp_buffer1->getAddress());
	kernel2->setArg(1, temp_buffer2->getAddress());
	kernel2->setArg(2, imax_buffer->getAddress());
	kernel2->setArg(3, static_cast<int>(imax_offset));
#else
	kernel2->setArg(0, temp_buffer1);
	kernel2->setArg(1, temp_buffer2);
	kernel2->setArg(2, imax_buffer);
	kernel2->setArg(3, static_cast<int>(imax_offset));
#endif

	// Launches the epilogue kernel
	//auto global2 = std::vector<size_t>{db["WGS2"]};
	auto global2 = std::vector<size_t>{1};
	auto local2 = std::vector<uint32_t>{db["WGS2"]};
	kernel2->enqueue(global2, {static_cast<uint32_t>(db["WGS2"])});
}

// =================================================================================================
}	// namespace clblast
