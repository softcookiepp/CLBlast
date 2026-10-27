
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xscal class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level1/xscal.hpp"

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
Xscal::Xscal(const tart::DType& dtype, Queue& queue, EventPointer event, const std::string& name)
		: Routine(queue, event, name, {"Xaxpy"}, dtype, {},
							{
	#include "../../kernels-vk-inline/level1/xscal.glsl.inl"
	,
	#include "../../kernels-vk-inline/level1/xscal-fast.glsl.inl"
							}
,
		{"Xscal", "XscalFast"}
	)
{
}

// =================================================================================================

// The main routine
void Xscal::DoScal(const size_t n, const tart::Scalar& alpha, const tart::buffer_ptr& x_buffer, const size_t x_offset,
											const size_t x_inc)
{
	// Makes sure all dimensions are larger than zero
	if (n == 0)
	{
		throw BLASError(StatusCode::kInvalidDimension);
	}

	// Tests the vector for validity
	TestVectorX(mDtype, n, x_buffer, x_offset, x_inc);

	// Determines whether or not the fast-version can be used
	bool use_fast_kernel = (x_offset == 0) && (x_inc == 1) && IsMultiple(n, db_["WGS"] * db_["WPT"] * db_["VW"]);

	// If possible, run the fast-version of the kernel
	auto kernel_name = (use_fast_kernel) ? "XscalFast" : "Xscal";
	
	// Retrieves the Xscal kernel from the compiled binary
	auto kernelOld = Kernel(program_, kernel_name);
	tart::kernel_ptr kernel = kernelOld.get();
	
#if VULKAN_API
	#if VULKAN_USE_BDA
	tart::DeviceMetadata meta = device_()->getMetadata();
	if (meta.bda)
	{
		if (use_fast_kernel)
		{
			kernel->setArg(0, static_cast<int>(n));
			kernel->setArgAs(mDtype, 1, (alpha));
			kernel->setArg(2, x_buffer->getAddress());
		}
		else
		{
			kernel->setArg(0, static_cast<int>(n));
			kernel->setArgAs(mDtype, 1, (alpha));
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
			kernel->setArgAs(mDtype, 1, (alpha));
			kernel->setArg(2, x_buffer);
		}
		else
		{
			kernel->setArg(0, static_cast<int>(n));
			kernel->setArgAs(mDtype, 1, (alpha));
			kernel->setArg(2, x_buffer);
			kernel->setArg(3, static_cast<int>(x_offset));
			kernel->setArg(4, static_cast<int>(x_inc));
		}
	}
#else
	// Sets the kernel arguments
	if (use_fast_kernel) {
		kernel->setArg(0, static_cast<int>(n));
		kernel->setArgAs(mDtype, 1, (alpha));
		kernel->setArg(2, x_buffer);
	} else {
		kernel->setArg(0, static_cast<int>(n));
		kernel->setArgAs(mDtype, 1, (alpha));
		kernel->setArg(2, x_buffer);
		kernel->setArg(3, static_cast<int>(x_offset));
		kernel->setArg(4, static_cast<int>(x_inc));
	}
#endif
	
	// Launches the kernel
	std::vector<uint32_t> global(3, 1);
	if (use_fast_kernel)
	{
		global[0] = CeilDiv(n, db_["WPT"] * db_["VW"]) / db_["WGS"];
	}
	else
	{
		auto n_ceiled = Ceil(n, db_["WGS"] * db_["WPT"]);
		global[0] = n_ceiled / (db_["WPT"] * db_["WGS"]);
	}
	kernel->enqueue(global, {db_["WGS"], db_["WPT"]});
	
}

// =================================================================================================
}	// namespace clblast
