
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the functional Xcopy routine.
//
// =================================================================================================

#include "routines/level1/xcopy.hpp"
#include "routine_aot_cache.hpp"
#include <cstddef>
#include <vector>

#include "routines/common.hpp"
#include "utilities/backend.hpp"
#include "utilities/buffer_test.hpp"
#include "utilities/clblast_exceptions.hpp"
#include "utilities/utilities.hpp"

namespace clblast {

void DoCopy(const tart::DType& dtype, const tart::device_ptr& device,
            const uint32_t n, const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
            const tart::buffer_ptr& y_buffer, const size_t y_offset, const size_t y_inc,
            const Databases& db)
{
	if (n == 0) {
		throw BLASError(StatusCode::kInvalidDimension);
	}

	TestVectorX(dtype, n, x_buffer, x_offset, x_inc);
	TestVectorY(dtype, n, y_buffer, y_offset, y_inc);

	bool use_fast_kernel = (x_offset == 0) && (x_inc == 1) && (y_offset == 0) && (y_inc == 1) &&
												 IsMultiple(n, db["WGS"] * db["WPT"] * db["VW"]);
	
	tart::kernel_ptr kernel = (use_fast_kernel) ? getCopyFast(device, dtype, db["VW"]) : getCopy(device, dtype, db["VW"]);
	
	if (use_fast_kernel)
	{
		kernel->setArg(0, static_cast<int>(n));
		kernel->setArg(1, x_buffer);
		kernel->setArg(2, y_buffer);
	}
	else
	{
		kernel->setArg(0, static_cast<int>(n));
		kernel->setArg(1, x_buffer);
		kernel->setArg(2, static_cast<int>(x_offset));
		kernel->setArg(3, static_cast<int>(x_inc));
		kernel->setArg(4, y_buffer);
		kernel->setArg(5, static_cast<int>(y_offset));
		kernel->setArg(6, static_cast<int>(y_inc));
	}

	std::vector<size_t> global(3, 1);
	if (use_fast_kernel)
	{
		global[0] = CeilDiv(n, db["WPT"] * db["VW"]) / db["WGS"];
	}
	else
	{
		global[0] = Ceil(n, db["WGS"] * db["WPT"]) / (db["WGS"] * db["WPT"]);
	}
	kernel->enqueue(global, {static_cast<uint32_t>(db["WGS"]), static_cast<uint32_t>(db["WPT"])});
}

}	// namespace clblast
