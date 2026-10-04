
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

// Enqueues a kernel, waits for completion, and checks for errors
void RunKernel(Kernel& kernel, Queue& queue, const Device& device, std::vector<size_t> global,
	const std::vector<size_t>& local)
{
	if (!local.empty())
	{
		auto local_size = size_t{1};
		for (auto& item : local) {
			local_size *= item;
		}

		// Verify that the global thread sizes are a multiple of the local sizes
		for (auto i = size_t{0}; i < global.size(); ++i) {
			if ((global[i] / local[i]) * local[i] != global[i]) {
				throw RuntimeErrorCode(StatusCode::kInvalidLocalThreadsDim,
															 ToString(global[i]) + " is not divisible by " + ToString(local[i]));
			}
		}
	}

// Prints the name of the kernel to launch in case of debugging in verbose mode
#ifdef VERBOSE
	queue.Finish();
	printf("[DEBUG] Running kernel '%s'\n", kernel.GetFunctionName().c_str());
	const auto start_time = std::chrono::steady_clock::now();
#endif

	// Launches the kernel (and checks for launch errors)
	
	kernel.Launch(queue, global, local);

// Prints the elapsed execution time in case of debugging in verbose mode
#ifdef VERBOSE
	queue.Finish();
	const auto elapsed_time = std::chrono::steady_clock::now() - start_time;
	const auto timing = std::chrono::duration<double, std::milli>(elapsed_time).count();
	printf("[DEBUG] Completed kernel in %.2lf ms\n", timing);
#endif
}

// =================================================================================================

// Sets all elements of a matrix to a constant value
template <typename T>
void FillMatrix(const std::shared_ptr<Program> program, const size_t m, const size_t n, const size_t ld,
								const size_t offset, const Buffer<T>& dest, const T constant_value, const size_t local_size) {
	Kernel kernelOld(program, "FillMatrix");
	tart::kernel_ptr kernel = kernelOld.get();
	kernel->setArg(0, static_cast<int>(m));
	kernel->setArg(1, static_cast<int>(n));
	kernel->setArg(2, static_cast<int>(ld));
	kernel->setArg(3, static_cast<int>(offset));
	kernel->setArg(4, dest());
	kernel->setArg(5, GetRealArg(constant_value));
	auto local = std::vector<uint32_t>{local_size, 1};
	auto global = std::vector<uint32_t>{Ceil(m, local_size) / local_size, n};
	kernel->enqueue(global, {});
}

// Compiles the above function
template void FillMatrix<half>(const std::shared_ptr<Program>, const size_t, const size_t, const size_t, const size_t,
	const Buffer<half>&, const half, const size_t);
template void FillMatrix<float>(const std::shared_ptr<Program>, const size_t, const size_t, const size_t, const size_t,
	const Buffer<float>&, const float, const size_t);
template void FillMatrix<double>(const std::shared_ptr<Program>, const size_t, const size_t, const size_t, const size_t,
	const Buffer<double>&, const double, const size_t);
template void FillMatrix<float2>(const std::shared_ptr<Program>, const size_t, const size_t, const size_t, const size_t,
	const Buffer<float2>&, const float2, const size_t);
template void FillMatrix<double2>(const std::shared_ptr<Program>, const size_t, const size_t, const size_t, const size_t,
	const Buffer<double2>&, const double2, const size_t);

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
	kernel->setArg(4, GetRealArg(constant_value));
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
