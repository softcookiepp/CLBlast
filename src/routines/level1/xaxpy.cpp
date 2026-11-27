
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the functional Xaxpy routine.
//
// =================================================================================================

#include "routines/level1/xaxpy.hpp"
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

// Functional implementation of Xaxpy (y = alpha*x + y)
void DoAxpy(const tart::DType& dtype, [[maybe_unused]] const tart::device_ptr& device,
            const uint32_t n, const tart::Scalar& alpha,
            const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
            const tart::buffer_ptr& y_buffer, const size_t y_offset, const size_t y_inc,
            const Databases& db)
{
	// Makes sure all dimensions are larger than zero
	if (n == 0) {
		throw BLASError(StatusCode::kInvalidDimension);
	}

	// Tests the vectors for validity
	TestVectorX(dtype, n, x_buffer, x_offset, x_inc);
	TestVectorY(dtype, n, y_buffer, y_offset, y_inc);

	// Determines whether or not the fast-version can be used
	bool use_faster_kernel = (x_offset == 0) && (x_inc == 1) && (y_offset == 0) && (y_inc == 1) && IsMultiple(n, db["WPT"] * db["VW"]);
	bool use_fastest_kernel = use_faster_kernel && IsMultiple(n, db["WGS"] * db["WPT"] * db["VW"]);
	use_faster_kernel = false;
	use_fastest_kernel = false;

	// If possible, run the fast-version of the kernel
	const auto kernel_name = (use_fastest_kernel) ? "XaxpyFastest" : (use_faster_kernel) ? "XaxpyFaster" : "Xaxpy";

	// Load kernel from AOT cache
	tart::kernel_ptr kernel = nullptr;
	if (use_fastest_kernel)
		kernel = getAxpyFastest(device, dtype, db["VW"]);
	else if (use_faster_kernel)
		kernel = getAxpyFaster(device, dtype, db["VW"]);
	else
		kernel = getAxpy(device, dtype, db["VW"]);

	// Sets the kernel arguments
	#if VULKAN_USE_BDA
	tart::DeviceMetadata meta = device->getMetadata();
	if (meta.bda)
	{
		if (use_faster_kernel || use_fastest_kernel)
		{
			if (!use_fastest_kernel) kernel->setArg(0, static_cast<int>(n));
			kernel->setArgAs(dtype, 1, alpha);
			kernel->setArg(2, x_buffer->getAddress());
			kernel->setArg(3, y_buffer->getAddress());
		} else {
			kernel->setArg(0, static_cast<int>(n));
			kernel->setArgAs(dtype, 1, alpha);
			kernel->setArg(2, x_buffer->getAddress() );
			kernel->setArg(3, static_cast<int>(x_offset));
			kernel->setArg(4, static_cast<int>(x_inc));
			kernel->setArg(5, y_buffer->getAddress() );
			kernel->setArg(6, static_cast<int>(y_offset));
			kernel->setArg(7, static_cast<int>(y_inc));
		}
	}
	else
	#endif
	{
		if (use_faster_kernel || use_fastest_kernel)
		{
			if (!use_fastest_kernel) kernel->setArg(0, static_cast<int>(n));
			kernel->setArgAs(dtype, 1, alpha);
			kernel->setArg(2, x_buffer);
			kernel->setArg(3, y_buffer);
		} else {
			kernel->setArg(0, static_cast<int>(n));
			kernel->setArgAs(dtype, 1, alpha);
			kernel->setArg(2, x_buffer);
			kernel->setArg(3, static_cast<int>(x_offset));
			kernel->setArg(4, static_cast<int>(x_inc));
			kernel->setArg(5, y_buffer);
			kernel->setArg(6, static_cast<int>(y_offset));
			kernel->setArg(7, static_cast<int>(y_inc));
		}
	}

	// Launches the kernel
	std::vector<size_t> global(3, 1);
	if (use_fastest_kernel)
	{
		global[0] = CeilDiv(n, db["WPT"] * db["VW"])/db["WGS"];
	}
	else if (use_faster_kernel)
	{
		global[0] = Ceil(CeilDiv(n, db["WPT"] * db["VW"]), db["WGS"])/db["WGS"];
	}
	else
	{
		const auto n_ceiled = Ceil(n, db["WGS"] * db["WPT"])/db["WGS"];
		global[0] = n_ceiled / db["WPT"];
	}
	kernel->enqueue(global, {static_cast<uint32_t>(db["WGS"]), static_cast<uint32_t>(db["WPT"])});
}

// =================================================================================================
}	// namespace clblast
