
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

#include "utilities/backend.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// Compiles a program from source code
std::shared_ptr<Program> CompileFromSource(const std::string& source_string, const tart::DType& dtype,
		const std::string& routine_name, const Device& device,
		std::vector<std::string>& options,
		const size_t run_preprocessor,	// 0: platform dependent, 1: always, 2: never
		const bool silent, std::map<std::string, std::string>& kernelSources)
{
	auto header_string = std::string{""};
	header_string += "#define DTYPE " + std::to_string(static_cast<uint32_t>(dtype())) + "\n";

	// Adds the name of the routine as a define
	//header_string += "#define ROUTINE_" + routine_name + " 1\n";

	// For specific devices, use staggered/shuffled workgroup indices.
	if (device()->getMetadata().physicalDeviceProperties.vendorID == tart::VendorID::eAMD)
	{
		header_string += "#define USE_STAGGERED_INDICES 1\n";
	}

	tart::DeviceMetadata meta = device()->getMetadata();
	if (meta.subgroupAdd)
	{
		header_string += "#define SUBGROUP_OPERATIONS_SUPPORTED 1\n";
	}
	else
	{
		header_string += "#define SUBGROUP_OPERATIONS_SUPPORTED 0\n";
		header_string += ("#define SUBGROUP_SIZE " + std::to_string(meta.subgroupSize) + "\n");
	}
	#if VULKAN_USE_BDA
		if (meta.bda)
		{
			// buffer device address support
			header_string += "#define USE_BDA 1\n";
		}
	#endif
	
	auto kernel_string = header_string + source_string;

	// Compiles the kernel
	std::shared_ptr<Program> program = nullptr;
	// append the header to everything
	for (auto& pair : kernelSources)
	{
		pair.second = "#version 450\n\n" + header_string + pair.second;
	}
	program = std::make_shared<Program>(device(), kernelSources);
	return program;
}
#if VULKAN_API
std::shared_ptr<Program> CompileFromSource(const std::string& source_string, const tart::DType& dtype,
                                           const std::string& routine_name, const Device& device,
                                           std::vector<std::string>& options,
                                           const size_t run_preprocessor,  // 0: platform dependent, 1: always, 2: never
                                           const bool silent)
{
	std::map<std::string, std::string> dummyMap;
	return CompileFromSource(source_string, dtype, routine_name, device,
                                           options,
                                           run_preprocessor,
                                            silent, dummyMap);
}
#endif
// =================================================================================================
}	// namespace clblast
