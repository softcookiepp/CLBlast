
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xhad class (see the header for information about the class).
//
// =================================================================================================

#include "routines/levelx/xhad.hpp"

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
Xhad::Xhad(const tart::DType& dtype, Queue& queue, EventPointer event, const std::string& name)
		: Routine(queue, event, name, {"Xaxpy"}, dtype, {},
							{
	#include "../../kernels-vk-inline/level1/xhad.glsl.inl"
	,
	#include "../../kernels-vk-inline/level1/xhad_faster.glsl.inl"
	,
	#include "../../kernels-vk-inline/level1/xhad_fastest.glsl.inl"
							}
,
 {"Xhad", "XhadFaster", "XhadFastest"}
	) {
}

// =================================================================================================

// The main routine
void Xhad::DoHad(const size_t n, const tart::Scalar& alpha, const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
	const tart::buffer_ptr& y_buffer, const size_t y_offset, const size_t y_inc, const tart::Scalar& beta,
	const tart::buffer_ptr& z_buffer, const size_t z_offset, const size_t z_inc)
{
	// Makes sure all dimensions are larger than zero
	if (n == 0) {
		throw BLASError(StatusCode::kInvalidDimension);
	}

	// Tests the vectors for validity
	TestVectorX(mDtype, n, x_buffer, x_offset, x_inc);
	TestVectorY(mDtype, n, y_buffer, y_offset, y_inc);
	TestVectorZ(mDtype, n, z_buffer, z_offset, z_inc);

	// Determines whether or not the fast-version can be used
	const auto use_faster_kernel = (x_offset == 0) && (x_inc == 1) && (y_offset == 0) && (y_inc == 1) &&
																 (z_offset == 0) && (z_inc == 1) && IsMultiple(n, db_["WPT"] * db_["VW"]);
	const auto use_fastest_kernel = use_faster_kernel && IsMultiple(n, db_["WGS"] * db_["WPT"] * db_["VW"]);

	// Retrieves the Xhad kernel from the compiled binary
	tart::kernel_ptr kernel = (use_fastest_kernel) ? getHadFastest(mDevice, mDtype, db_["VW"]) : (use_faster_kernel) ? getHadFaster(mDevice, mDtype, db_["VW"]) : getHad(mDevice, mDtype, db_["VW"]);
	
	// Sets the kernel arguments
	if (use_faster_kernel || use_fastest_kernel) {
		kernel->setArg(0, static_cast<int>(n));
		kernel->setArgAs(mDtype, 1, alpha);
		kernel->setArgAs(mDtype, 2, (beta));
		kernel->setArg(3, x_buffer);
		kernel->setArg(4, y_buffer);
		kernel->setArg(5, z_buffer);
	} else {
		kernel->setArg(0, static_cast<int>(n));
		kernel->setArgAs(mDtype, 1, (alpha));
		kernel->setArgAs(mDtype, 2, (beta));
		kernel->setArg(3, x_buffer);
		kernel->setArg(4, static_cast<int>(x_offset));
		kernel->setArg(5, static_cast<int>(x_inc));
		kernel->setArg(6, y_buffer);
		kernel->setArg(7, static_cast<int>(y_offset));
		kernel->setArg(8, static_cast<int>(y_inc));
		kernel->setArg(9, z_buffer);
		kernel->setArg(10, static_cast<int>(z_offset));
		kernel->setArg(11, static_cast<int>(z_inc));
	}

	// Launches the kernel
	std::vector<uint32_t> global(3, 1);
	if (use_fastest_kernel)
	{
		global[0] = CeilDiv(n, db_["WPT"] * db_["VW"]) / db_["WGS"];
	}
	else if (use_faster_kernel)
	{
		global[0] = Ceil(CeilDiv(n, db_["WPT"] * db_["VW"]), db_["WGS"]) / db_["WGS"];
	}
	else
	{
		global[0] = Ceil(n, db_["WGS"] * db_["WPT"]) / db_["WPT"];
	}
	kernel->enqueue(global, {db_["WGS"], db_["WPT"]});
}

// =================================================================================================
}	// namespace clblast
