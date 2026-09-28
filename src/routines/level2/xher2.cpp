
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xher2 class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level2/xher2.hpp"

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
Xher2<T>::Xher2(Queue& queue, EventPointer event, const std::string& name)
		: Routine(queue, event, name, {"Xger"}, PrecisionValue<T>(), {},
							{
	#include "../../kernels-vk-inline/level2/xher2.glsl.inl"
							}
,
 {"Xher2"}
		)
{
	mHPR = static_cast<uint32_t>(name == "HPR");
	mSPR = static_cast<uint32_t>(name == "SPR");
	mGERC = static_cast<uint32_t>(name == "GERC");
	mHER = static_cast<uint32_t>(name == "HER");
	mHER2 = static_cast<uint32_t>(name == "HER2");
	mHPR2 = static_cast<uint32_t>(name == "HPR2");
	mSPR2 = static_cast<uint32_t>(name == "SPR2");
}

// =================================================================================================

// The main routine
template <typename T>
void Xher2<T>::DoHer2(const Layout layout, const Triangle triangle, const size_t n, const T alpha,
											const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc, const Buffer<T>& y_buffer,
											const size_t y_offset, const size_t y_inc, const Buffer<T>& a_buffer, const size_t a_offset,
											const size_t a_ld, const bool packed)
{
	// Makes sure the dimensions are larger than zero
	if (n == 0) {
		throw BLASError(StatusCode::kInvalidDimension);
	}

	// The data is either in the upper or lower triangle
	const auto is_upper = ((triangle == Triangle::kUpper && layout != Layout::kRowMajor) ||
												 (triangle == Triangle::kLower && layout == Layout::kRowMajor));
	const auto is_rowmajor = (layout == Layout::kRowMajor);

	// Tests the matrix and the vectors for validity
	if (packed) {
		TestMatrixAP(n, a_buffer, a_offset);
	} else {
		TestMatrixA(n, n, a_buffer, a_offset, a_ld);
	}
	TestVectorX(n, x_buffer, x_offset, x_inc);
	TestVectorY(n, y_buffer, y_offset, y_inc);

	// Retrieves the kernel from the compiled binary
	auto kernelOld = Kernel(program_, "Xher2");
	tart::kernel_ptr kernel = kernelOld.get();
	
	// Sets the kernel arguments
	kernel->setArg(0, static_cast<int>(n));
	kernel->setArg(1, GetRealArg(alpha));
	kernel->setArg(2, x_buffer());
	kernel->setArg(3, static_cast<int>(x_offset));
	kernel->setArg(4, static_cast<int>(x_inc));
	kernel->setArg(5, y_buffer());
	kernel->setArg(6, static_cast<int>(y_offset));
	kernel->setArg(7, static_cast<int>(y_inc));
	kernel->setArg(8, a_buffer());
	kernel->setArg(9, static_cast<int>(a_offset));
	kernel->setArg(10, static_cast<int>(a_ld));
	kernel->setArg(11, static_cast<int>(is_upper));
	kernel->setArg(12, static_cast<int>(is_rowmajor));

	// Launches the kernel
	auto global_one = Ceil(CeilDiv(n, db_["WPT"]), db_["WGS1"]);
	auto global_two = Ceil(CeilDiv(n, db_["WPT"]), db_["WGS2"]);
	auto global = std::vector<uint32_t>{global_one / db_["WGS1"], global_two / db_["WGS2"]};
	auto local = std::vector<size_t>{db_["WGS1"], db_["WGS2"]};
	//RunKernel(kernel, queue_, device_, global, local);
	kernel->enqueue(global, {db_["WGS1"], db_["WGS2"], db_["WPT"],
		mHPR,
		mSPR,
		mGERC,
		mHER,
		mHER2,
		mHPR2,
		mSPR2
	});
}

// =================================================================================================

// Compiles the templated class
template class Xher2<half>;
template class Xher2<float>;
template class Xher2<double>;
template class Xher2<float2>;
template class Xher2<double2>;

// =================================================================================================
}	// namespace clblast
