
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xdot class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level1/xdot.hpp"

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
Xdot<T>::Xdot(Queue& queue, EventPointer event, const std::string& name)
		: Routine(queue, event, name, {"Xdot"}, PrecisionValue<T>(), {},
							{
	#include "../../kernels-vk-inline/level1/xdot.glsl.inl"
	,
	#include "../../kernels-vk-inline/level1/xdot_epilogue.glsl.inl"
							}
,
	 {"Xdot", "XdotEpilogue"}
	)
{
}

// =================================================================================================

// The main routine
template <typename T>
void Xdot<T>::DoDot(const size_t n, const Buffer<T>& dot_buffer, const size_t dot_offset, const Buffer<T>& x_buffer,
										const size_t x_offset, const size_t x_inc, const Buffer<T>& y_buffer, const size_t y_offset,
										const size_t y_inc, const bool do_conjugate) {
	// Makes sure all dimensions are larger than zero
	if (n == 0) {
		throw BLASError(StatusCode::kInvalidDimension);
	}

	// Tests the vectors for validity
	TestVectorX(n, x_buffer, x_offset, x_inc);
	TestVectorY(n, y_buffer, y_offset, y_inc);
	TestVectorScalar(1, dot_buffer, dot_offset);

	// Retrieves the Xdot kernels from the compiled binary
	auto kernel1Old = Kernel(program_, "Xdot");
	tart::kernel_ptr kernel1 = kernel1Old.get();
	auto kernel2Old = Kernel(program_, "XdotEpilogue");
	tart::kernel_ptr kernel2 = kernel2Old.get();

	// Creates the buffer for intermediate values
	auto temp_size = 2 * db_["WGS2"];
	auto temp_buffer = Buffer<T>(queue_(), temp_size);
	
	// Sets the kernel arguments
#if VULKAN_USE_BDA
	tart::DeviceMetadata meta = device_()->getMetadata();
	if (meta.bda)
	{
		kernel1->setArg(0, static_cast<int>(n));
		kernel1->setArg(1, x_buffer()->getAddress());
		kernel1->setArg(2, static_cast<int>(x_offset));
		kernel1->setArg(3, static_cast<int>(x_inc));
		kernel1->setArg(4, y_buffer()->getAddress());
		kernel1->setArg(5, static_cast<int>(y_offset));
		kernel1->setArg(6, static_cast<int>(y_inc));
		kernel1->setArg(7, temp_buffer()->getAddress());
		kernel1->setArg(8, static_cast<int>(do_conjugate));
	}
	else
#endif
	{
		kernel1->setArg(0, static_cast<int>(n));
		kernel1->setArg(1, x_buffer());
		kernel1->setArg(2, static_cast<int>(x_offset));
		kernel1->setArg(3, static_cast<int>(x_inc));
		kernel1->setArg(4, y_buffer());
		kernel1->setArg(5, static_cast<int>(y_offset));
		kernel1->setArg(6, static_cast<int>(y_inc));
		kernel1->setArg(7, temp_buffer());
		kernel1->setArg(8, static_cast<int>(do_conjugate));
	}

	// Launches the main kernel
	auto global1 = std::vector<size_t>{db_["WGS1"] * temp_size};
	auto local1 = std::vector<size_t>{db_["WGS1"]};
	kernel1->setArg(9, static_cast<int>(temp_size));
	kernel1->enqueue({temp_size}, {db_["WGS1"], 1});
	
	// Sets the arguments for the epilogue kernel
	#if VULKAN_USE_BDA
	if (meta.bda)
	{
		kernel2->setArg(0, temp_buffer()->getAddress());
		kernel2->setArg(1, dot_buffer()->getAddress());
		kernel2->setArg(2, static_cast<int>(dot_offset));
	}
	else
	#endif
	{
		kernel2->setArg(0, temp_buffer());
		kernel2->setArg(1, dot_buffer());
		kernel2->setArg(2, static_cast<int>(dot_offset));
	}

	// Launches the epilogue kernel
	auto global2 = std::vector<size_t>{db_["WGS2"]};
	auto local2 = std::vector<size_t>{db_["WGS2"]};
	kernel2->enqueue({1}, {db_["WGS2"], 1});
	
}

// =================================================================================================

// Compiles the templated class
template class Xdot<half>;
template class Xdot<float>;
template class Xdot<double>;
template class Xdot<float2>;
template class Xdot<double2>;

// =================================================================================================
}	// namespace clblast
