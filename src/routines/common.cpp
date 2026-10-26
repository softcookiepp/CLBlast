
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the common routine functions (see the header for more information).
//
// =================================================================================================

#include "routines/common.hpp"

#include <cstddef>
#include <memory>
#include <vector>

#include "utilities/backend.hpp"
#include "utilities/clblast_exceptions.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// Sets all elements of a matrix to a constant value
void FillMatrix(const tart::DType& dtype, const std::shared_ptr<Program> program, const size_t m, const size_t n, const size_t ld,
	const size_t offset, const tart::buffer_ptr& dest, const tart::Scalar constant_value, const size_t local_size)
{
	Kernel kernelOld(program, "FillMatrix");
	tart::kernel_ptr kernel = kernelOld.get();
	kernel->setArg(0, static_cast<int>(m));
	kernel->setArg(1, static_cast<int>(n));
	kernel->setArg(2, static_cast<int>(ld));
	kernel->setArg(3, static_cast<int>(offset));
	kernel->setArg(4, dest);
	kernel->setArgAs(dtype, 5, constant_value);
	auto local = std::vector<uint32_t>{local_size, 1};
	auto global = std::vector<uint32_t>{Ceil(m, local_size) / local_size, n};
	kernel->enqueue(global, {});
}

// Sets all elements of a vector to a constant value
template <typename T>
void FillVector(const std::shared_ptr<Program> program, const size_t n, const size_t inc, const size_t offset,
								const Buffer<T>& dest, const T constant_value, const size_t local_size) {
	Kernel kernelOld(program, "FillVector");
	tart::kernel_ptr kernel = kernelOld.get();
	kernel->setArg(0, static_cast<int>(n));
	kernel->setArg(1, static_cast<int>(inc));
	kernel->setArg(2, static_cast<int>(offset));
	kernel->setArg(3, dest());
	kernel->setArgAs(tart::getDType<T>(), 4, (constant_value));
	auto local = std::vector<uint32_t>{local_size};
	auto global = std::vector<uint32_t>{Ceil(n, local_size) / local_size};
	kernel->enqueue(global, {});
}

// Compiles the above function
template void FillVector<half>(const std::shared_ptr<Program>, const size_t, const size_t, const size_t, const Buffer<half>&,
	const half, const size_t);
template void FillVector<float>(const std::shared_ptr<Program>, const size_t, const size_t, const size_t,
	const Buffer<float>&, const float, const size_t);
template void FillVector<double>(const std::shared_ptr<Program>, const size_t, const size_t, const size_t,
	const Buffer<double>&, const double, const size_t);
template void FillVector<float2>(const std::shared_ptr<Program>, const size_t, const size_t, const size_t,
	const Buffer<float2>&, const float2, const size_t);
template void FillVector<double2>(const std::shared_ptr<Program>, const size_t, const size_t, const size_t,
	const Buffer<double2>&, const double2, const size_t);

// =================================================================================================
}	// namespace clblast
