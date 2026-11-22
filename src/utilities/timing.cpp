
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file provides helper functions for time measurement and such.
//
// =================================================================================================

#include "utilities/timing.hpp"

#include <cstdio>
#include <utility>
#include <vector>

#include "utilities/backend.hpp"
#include "utilities/clblast_exceptions.hpp"

namespace clblast {
// =================================================================================================

double RunKernelTimed(const uint32_t num_runs, const tart::kernel_ptr& kernel, Queue& queue, const Device& device,
											std::vector<uint32_t> global, const std::vector<uint32_t>& local,
											const std::vector<uint32_t>& spec) {
	if (!local.empty()) {
		// Tests for validity of the local thread sizes
		auto local_size = uint32_t{1};
		for (auto& item : local) {
			local_size *= item;
		}

		// Make sure the global thread sizes are at least equal to the local sizes
		for (auto i = uint32_t{0}; i < global.size(); ++i) {
			if (global[i] < local[i]) {
				global[i] = local[i];
			}
		}
	}
	
	// finish pre-emptively so time is accurate
	queue.Finish();

	// Times the kernel
	const auto run_kernel_func = [&]()
	{
		kernel->enqueue(global, local);
		queue()->sync();
	};
	return TimeFunction(num_runs, run_kernel_func);
}

double TimeKernel(const uint32_t num_runs, const tart::kernel_ptr& kernel, Queue& queue, const Device& device, std::vector<uint32_t> global,
	const std::vector<uint32_t>& local, const bool silent, const std::vector<uint32_t>& spec)
{
	const auto time_ms = RunKernelTimed(num_runs, kernel, queue, device, std::move(global), local);
	if (!silent)
	{
		printf(" %9.2lf ms |", time_ms);
	}
	return time_ms;
}

// =================================================================================================
}	// namespace clblast
