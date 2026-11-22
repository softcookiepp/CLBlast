
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the functional Xnrm2 routine.
//
// =================================================================================================

#include "routines/level1/xnrm2.hpp"
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

// Functional implementation of Xnrm2 (Euclidean norm)
void DoNrm2(const tart::DType& dtype, const tart::device_ptr& device,
            const size_t n, const tart::buffer_ptr& nrm2_buffer, const size_t nrm2_offset,
            const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
            const Databases& db)
{
	// Makes sure all dimensions are larger than zero
	if (n == 0) {
		throw BLASError(StatusCode::kInvalidDimension);
	}

	// Tests the vectors for validity
	TestVectorX(dtype, n, x_buffer, x_offset, x_inc);
	TestVectorScalar(dtype, 1, nrm2_buffer, nrm2_offset);

	// Retrieves the Xnrm2 kernels from the compiled binary
	tart::kernel_ptr kernel1 = getNrm2(device, dtype);
	tart::kernel_ptr kernel2 = getNrm2Epilogue(device, dtype);

	// Creates the buffer for intermediate values
	auto temp_size = 2 * db["WGS2"];
	auto temp_buffer = device->allocateBuffer(temp_size*dtype.size());

	// Sets the kernel arguments
	kernel1->setArg(0, static_cast<int>(n));
	kernel1->setArg(1, x_buffer);
	kernel1->setArg(2, static_cast<int>(x_offset));
	kernel1->setArg(3, static_cast<int>(x_inc));
	kernel1->setArg(4, temp_buffer);

	// Launches the main kernel
	kernel1->enqueue({temp_size}, {db["WGS1"], 1});
	
	// Sets the arguments for the epilogue kernel
	kernel2->setArg(0, temp_buffer);
	kernel2->setArg(1, nrm2_buffer);
	kernel2->setArg(2, static_cast<int>(nrm2_offset));

	// Launches the epilogue kernel
	kernel2->enqueue({temp_size}, {db["WGS2"], 1});
}

// =================================================================================================
}	// namespace clblast
