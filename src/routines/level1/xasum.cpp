
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xasum class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level1/xasum.hpp"

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
template <typename T>
Xasum<T>::Xasum(Queue& queue, EventPointer event, const std::string& name)
		: Routine(queue, event, name, {"Xdot"}, PrecisionValue<T>(), {},
							{
#if VULKAN_API
	#include "../../kernels-vk-inline/level1/xasum.glsl.inl"
	,
	#include "../../kernels-vk-inline/level1/xasum_epilogue.glsl.inl"
#else
	#include "../../kernels/level1/xasum.opencl"
#endif
							}
,
		
			{"Xasum", "XasumEpilogue"}
			)							
{
	mSum = static_cast<uint32_t>(name == "SUM");
}

// =================================================================================================

// The main routine
template <typename T>
void Xasum<T>::DoAsum(const size_t n, const Buffer<T>& asum_buffer, const size_t asum_offset, const Buffer<T>& x_buffer,
											const size_t x_offset, const size_t x_inc)
{
	// Makes sure all dimensions are larger than zero
	if (n == 0)
	{
		throw BLASError(StatusCode::kInvalidDimension);
	}

	// Tests the vectors for validity
	TestVectorX(n, x_buffer, x_offset, x_inc);
	TestVectorScalar(1, asum_buffer, asum_offset);

	// Retrieves the Xasum kernels from the compiled binary
	auto kernel1Old = Kernel(program_, "Xasum");
	tart::kernel_ptr kernel1 = kernel1Old.get();
	auto kernel2Old = Kernel(program_, "XasumEpilogue");
	tart::kernel_ptr kernel2 = kernel2Old.get();
	
	// Creates the buffer for intermediate values
	auto temp_size = 2 * db_["WGS2"];
	auto temp_buffer = Buffer<T>(queue_(), temp_size);

	// Sets the kernel arguments
	#if VULKAN_USE_BDA
		tart::DeviceMetadata meta = device_()->getMetadata();
		kernel1->setArg(0, static_cast<int>(n));
		if (meta.bda)
			kernel1->setArg(1, x_buffer()->getAddress());
		else
			kernel1->setArg(1, x_buffer());
		kernel1->setArg(2, static_cast<int>(x_offset));
		kernel1->setArg(3, static_cast<int>(x_inc));
		if (meta.bda)
			kernel1->setArg(4, temp_buffer()->getAddress());
		else
			kernel1->setArg(4, temp_buffer());
	#else
		kernel1->setArg(0, static_cast<int>(n));
		kernel1->setArg(1, x_buffer());
		kernel1->setArg(2, static_cast<int>(x_offset));
		kernel1->setArg(3, static_cast<int>(x_inc));
		kernel1->setArg(4, temp_buffer());
	#endif

	// Launches the main kernel
	kernel1->enqueue({temp_size, 1, 1}, {db_["WGS1"], mSum});

	// Sets the arguments for the epilogue kernel
#if VULKAN_USE_BDA
	if (meta.bda)
	{
		kernel2->setArg(0, temp_buffer()->getAddress());
		kernel2->setArg(1, asum_buffer()->getAddress());
	}
	else
	{
		kernel2->setArg(0, temp_buffer());
		kernel2->setArg(1, asum_buffer());
	}
	kernel2->setArg(2, static_cast<int>(asum_offset));
#else
	kernel2->setArg(0, temp_buffer());
	kernel2->setArg(1, asum_buffer());
	kernel2->setArg(2, static_cast<int>(asum_offset));
#endif

	// Launches the epilogue kernel
	kernel2->enqueue({1, 1, 1}, {db_["WGS2"]});
}

// =================================================================================================

// Compiles the templated class
template class Xasum<half>;
template class Xasum<float>;
template class Xasum<double>;
template class Xasum<float2>;
template class Xasum<double2>;

// =================================================================================================
}	// namespace clblast
