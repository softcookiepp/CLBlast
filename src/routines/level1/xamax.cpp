
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xamax class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level1/xamax.hpp"

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

// Constructor: forwards to base class constructor
template <typename T>
Xamax<T>::Xamax(Queue& queue, EventPointer event, const std::string& name)
	: Routine(queue, event, name, {"Xdot"}, PrecisionValue<T>(), {},
		{
			#include "../../kernels-vk-inline/level1/xamax.glsl.inl"
			,
			#include "../../kernels-vk-inline/level1/xamax_epilogue.glsl.inl"
		},
		{"Xamax", "XamaxEpilogue"}
	)
{
	mMax = static_cast<uint32_t>(name == "MAX");
	mMin = static_cast<uint32_t>(name == "MIN");
	mAmin = static_cast<uint32_t>(name == "AMIN");
}

// =================================================================================================

// The main routine
template <typename T>
void Xamax<T>::DoAmax(const size_t n, const Buffer<unsigned int>& imax_buffer, const size_t imax_offset,
											const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc) {
	// Makes sure all dimensions are larger than zero
	if (n == 0) {
		throw BLASError(StatusCode::kInvalidDimension);
	}

	// Tests the vectors for validity
	TestVectorX(n, x_buffer, x_offset, x_inc);
	TestVectorIndex(1, imax_buffer, imax_offset);

	// Retrieves the Xamax kernels from the compiled binary
	auto kernel1Old = Kernel(program_, "Xamax");
	tart::kernel_ptr kernel1 = kernel1Old.get();
	auto kernel2Old = Kernel(program_, "XamaxEpilogue");
	tart::kernel_ptr kernel2 = kernel2Old.get();

	// Creates the buffer for intermediate values
	auto temp_size = 2 * db_["WGS2"];
	auto temp_buffer1 = Buffer<T>(mDevice, temp_size);
	auto temp_buffer2 = Buffer<unsigned int>(mDevice, temp_size);

	// Sets the kernel arguments
#if VULKAN_USE_BDA
	const tart::DeviceMetadata meta& = device_()->getMetadata();
	if (meta.bda)
	{
		kernel1->setArg(0, static_cast<int>(n));
		kernel1->setArg(1, x_buffer()->getAddress() + x_offset*sizeof(T));
		kernel1->setArg(2, static_cast<int>(0));
		kernel1->setArg(3, static_cast<int>(x_inc));
		kernel1->setArg(4, temp_buffer1()->getAddress());
		kernel1->setArg(5, temp_buffer2()->getAddress());
	}
	else
#endif
	{
		kernel1->setArg(0, static_cast<int>(n));
		kernel1->setArg(1, x_buffer());
		kernel1->setArg(2, static_cast<int>(x_offset));
		kernel1->setArg(3, static_cast<int>(x_inc));
		kernel1->setArg(4, temp_buffer1());
		kernel1->setArg(5, temp_buffer2());
	}

	// Launches the main kernel
	//auto global1 = std::vector<size_t>{db_["WGS1"] * temp_size};
	auto global1 = std::vector<uint32_t>{temp_size};
	//auto local1 = std::vector<size_t>{db_["WGS1"]};
	
	// the number of workgroups in the X dimension
	int num_groups_0 = static_cast<int>(global1[0]);
	kernel1->setArg(6, num_groups_0);
	
	kernel1->enqueue(global1, {db_["WGS1"], mMax, mMin, mAmin});

	// Sets the arguments for the epilogue kernel
#if VULKAN_USE_BDA
	kernel2->setArg(0, temp_buffer1()->getAddress());
	kernel2->setArg(1, temp_buffer2()->getAddress());
	kernel2->setArg(2, imax_buffer()->getAddress());
	kernel2->setArg(3, static_cast<int>(imax_offset));
#else
	kernel2->setArg(0, temp_buffer1());
	kernel2->setArg(1, temp_buffer2());
	kernel2->setArg(2, imax_buffer());
	kernel2->setArg(3, static_cast<int>(imax_offset));
#endif

	// Launches the epilogue kernel
	//auto global2 = std::vector<size_t>{db_["WGS2"]};
	auto global2 = std::vector<uint32_t>{1};
	auto local2 = std::vector<size_t>{db_["WGS2"]};
	kernel2->enqueue(global2, {db_["WGS2"]});
}

// =================================================================================================

// Compiles the templated class
template class Xamax<half>;
template class Xamax<float>;
template class Xamax<double>;
template class Xamax<float2>;
template class Xamax<double2>;

// =================================================================================================
}	// namespace clblast
