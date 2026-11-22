
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the functional Xscal routine.
//
// =================================================================================================

#include "routines/level1/xscal.hpp"
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

// Functional implementation of Xscal
void DoScal(const tart::DType& dtype, const tart::device_ptr& device,
            const size_t n, const tart::Scalar& alpha,
            const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
            const Databases& db)
{
	// Makes sure all dimensions are larger than zero
	if (n == 0)
	{
		throw BLASError(StatusCode::kInvalidDimension);
	}

	// Tests the vector for validity
	TestVectorX(dtype, n, x_buffer, x_offset, x_inc);

	// Determines whether or not the fast-version can be used
	bool use_fast_kernel = (x_offset == 0) && (x_inc == 1) && IsMultiple(n, db["WGS"] * db["WPT"] * db["VW"]);

	// If possible, run the fast-version of the kernel
	tart::kernel_ptr kernel = (use_fast_kernel) ? getScalFast(device, dtype, db["VW"]) : getScal(device, dtype, db["VW"]);
	
#if VULKAN_API
	#if VULKAN_USE_BDA
	tart::DeviceMetadata meta = device->getMetadata();
	if (meta.bda)
	{
		if (use_fast_kernel)
		{
			kernel->setArg(0, static_cast<int>(n));
			kernel->setArgAs(dtype, 1, (alpha));
			kernel->setArg(2, x_buffer->getAddress());
		}
		else
		{
			kernel->setArg(0, static_cast<int>(n));
			kernel->setArgAs(dtype, 1, (alpha));
			kernel->setArg(2, x_buffer->getAddress());
			kernel->setArg(3, static_cast<int>(x_offset));
			kernel->setArg(4, static_cast<int>(x_inc));
		}
	}
	else
	#endif
	{
		// Sets the kernel arguments
		if (use_fast_kernel)
		{
			kernel->setArg(0, static_cast<int>(n));
			kernel->setArgAs(dtype, 1, (alpha));
			kernel->setArg(2, x_buffer);
		}
		else
		{
			kernel->setArg(0, static_cast<int>(n));
			kernel->setArgAs(dtype, 1, (alpha));
			kernel->setArg(2, x_buffer);
			kernel->setArg(3, static_cast<int>(x_offset));
			kernel->setArg(4, static_cast<int>(x_inc));
		}
	}
#else
	// Sets the kernel arguments
	if (use_fast_kernel) {
		kernel->setArg(0, static_cast<int>(n));
		kernel->setArgAs(dtype, 1, (alpha));
		kernel->setArg(2, x_buffer);
	} else {
		kernel->setArg(0, static_cast<int>(n));
		kernel->setArgAs(dtype, 1, (alpha));
		kernel->setArg(2, x_buffer);
		kernel->setArg(3, static_cast<int>(x_offset));
		kernel->setArg(4, static_cast<int>(x_inc));
	}
#endif
	
	// Launches the kernel
	std::vector<uint32_t> global(3, 1);
	if (use_fast_kernel)
	{
		global[0] = CeilDiv(n, db["WPT"] * db["VW"]) / db["WGS"];
	}
	else
	{
		auto n_ceiled = Ceil(n, db["WGS"] * db["WPT"]);
		global[0] = n_ceiled / (db["WPT"] * db["WGS"]);
	}
	kernel->enqueue(global, {db["WGS"], db["WPT"]});
	
}

// =================================================================================================
}	// namespace clblast
