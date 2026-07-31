
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the kernel compilation functions (see the header for more information).
//
// =================================================================================================

#include <cstddef>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "kernel_preprocessor.hpp"
#include "utilities/backend.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// Compiles a program from source code
std::shared_ptr<Program> CompileFromSource(const std::string& source_string, const Precision precision,
		const std::string& routine_name, const Device& device,
		std::vector<std::string>& options,
		const size_t run_preprocessor,	// 0: platform dependent, 1: always, 2: never
		const bool silent
#if VULKAN_API
,
		std::map<std::string, std::string>& kernelSources
#endif
		)
{
	auto header_string = std::string{""};

	header_string += "#define PRECISION " + ToString(static_cast<int>(precision)) + "\n";

	// Adds the name of the routine as a define
	header_string += "#define ROUTINE_" + routine_name + "\n";

	// Just use this on every device, no point in not doing it
	header_string += "#define USE_CL_MAD 1\n";

	// For specific devices, use staggered/shuffled workgroup indices.
	if (device()->getMetadata().physicalDeviceProperties.vendorID == tart::VendorID::eAMD)
	{
		header_string += "#define USE_STAGGERED_INDICES 1\n";
	}

	tart::DeviceMetadata meta = device()->getMetadata();
	if (meta.subgroupShuffle)
	{
		header_string += "#define USE_SUBGROUP_SHUFFLING 1\n";
		header_string += ("#define SUBGROUP_SIZE " + std::to_string(meta.subgroupSize) + "\n");
	}
	#if VULKAN_USE_BDA
		if (meta.bda)
		{
			// buffer device address support
			header_string += "#define USE_BDA 1\n";
		}
	#endif

	header_string +=
#include "kernels-vk-inline/common.glsl.inl"
	;

// Prints details of the routine to compile in case of debugging in verbose mode
#ifdef VERBOSE
	printf("[DEBUG] Compiling routine '%s-%s'\n", routine_name.c_str(), ToString(precision).c_str());
	const auto start_time = std::chrono::steady_clock::now();
#endif

	// Runs a pre-processor to unroll loops and perform array-to-register promotion. Most OpenCL
	// compilers do this, but some don't.
	auto do_run_preprocessor = false;
	
	auto kernel_string = header_string + source_string;
	if (do_run_preprocessor) {
		log_debug("Running built-in pre-processor");
		kernel_string = PreprocessKernelSource(kernel_string);
	}
	// Compiles the kernel
	std::shared_ptr<Program> program = nullptr;
	if (true)
	{
		// append the header to everything
		for (auto& pair : kernelSources)
		{
			pair.second = "#version 450\n\n" + header_string + pair.second;
		}
		program = std::make_shared<Program>(device(), kernelSources);
	}

// Prints the elapsed compilation time in case of debugging in verbose mode
#ifdef VERBOSE
	const auto elapsed_time = std::chrono::steady_clock::now() - start_time;
	const auto timing = std::chrono::duration<double, std::milli>(elapsed_time).count();
	printf("[DEBUG] Completed compilation in %.2lf ms\n", timing);
#endif

	return program;
}
#if VULKAN_API
std::shared_ptr<Program> CompileFromSource(const std::string& source_string, const Precision precision,
                                           const std::string& routine_name, const Device& device,
                                           std::vector<std::string>& options,
                                           const size_t run_preprocessor,  // 0: platform dependent, 1: always, 2: never
                                           const bool silent)
{
	std::map<std::string, std::string> dummyMap;
	return CompileFromSource(source_string, precision, routine_name, device,
                                           options,
                                           run_preprocessor,
                                            silent, dummyMap);
}
#endif
// =================================================================================================
}	// namespace clblast
