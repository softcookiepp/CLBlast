
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xnrm2 class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level1/xnrm2.hpp"

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
Xnrm2<T>::Xnrm2(Queue& queue, EventPointer event, const std::string& name)
		: Routine(queue, event, name, {"Xdot"}, PrecisionValue<T>(), {},
							{
	#include "../../kernels-vk-inline/level1/xnrm2.glsl.inl"
	,
	#include "../../kernels-vk-inline/level1/xnrm2-epilogue.glsl.inl"
							}
,
 {"Xnrm2", "Xnrm2Epilogue"}
		) {
}

// =================================================================================================

// The main routine
template <typename T>
void Xnrm2<T>::DoNrm2(const size_t n, const Buffer<T>& nrm2_buffer, const size_t nrm2_offset, const Buffer<T>& x_buffer,
											const size_t x_offset, const size_t x_inc)
{
	// Makes sure all dimensions are larger than zero
	if (n == 0) {
		throw BLASError(StatusCode::kInvalidDimension);
	}

	// Tests the vectors for validity
	TestVectorX(n, x_buffer, x_offset, x_inc);
	TestVectorScalar(1, nrm2_buffer, nrm2_offset);

	// Retrieves the Xnrm2 kernels from the compiled binary
	auto kernel1Old = Kernel(program_, "Xnrm2");
	tart::kernel_ptr kernel1 = kernel1Old.get();
	auto kernel2Old = Kernel(program_, "Xnrm2Epilogue");
	tart::kernel_ptr kernel2 = kernel2Old.get();

	// Creates the buffer for intermediate values
	auto temp_size = 2 * db_["WGS2"];
	auto temp_buffer = Buffer<T>(queue_(), temp_size);

	// Sets the kernel arguments
	kernel1->setArg(0, static_cast<int>(n));
	kernel1->setArg(1, x_buffer());
	kernel1->setArg(2, static_cast<int>(x_offset));
	kernel1->setArg(3, static_cast<int>(x_inc));
	kernel1->setArg(4, temp_buffer());

	// Launches the main kernel
	kernel1->enqueue({temp_size}, {db_["WGS1"], 1});
	
	// Sets the arguments for the epilogue kernel
	kernel2->setArg(0, temp_buffer());
	kernel2->setArg(1, nrm2_buffer());
	kernel2->setArg(2, static_cast<int>(nrm2_offset));

	// Launches the epilogue kernel
	kernel2->enqueue({temp_size}, {db_["WGS2"], 1});
}

// =================================================================================================

// Compiles the templated class
template class Xnrm2<half>;
template class Xnrm2<float>;
template class Xnrm2<double>;
template class Xnrm2<float2>;
template class Xnrm2<double2>;

// =================================================================================================
}	// namespace clblast
