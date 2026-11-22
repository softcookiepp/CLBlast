
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the kernel compilation functions (see the header for more information).
//
// =================================================================================================

#include <cstddef>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "utilities/backend.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// Helper: build preprocessor defines header
static std::string BuildDefinesHeader(const tart::DType& dtype, const Device& device) {
	auto header_string = std::string{""};
	header_string += "#define DTYPE " + std::to_string(static_cast<uint32_t>(dtype())) + "\n";

	tart::DeviceMetadata meta = device()->getMetadata();
	if (meta.subgroupAdd) {
		header_string += "#define SUBGROUP_OPERATIONS_SUPPORTED 1\n";
	}
	else {
		header_string += "#define SUBGROUP_OPERATIONS_SUPPORTED 0\n";
		header_string += ("#define SUBGROUP_SIZE " + std::to_string(meta.subgroupSize) + "\n");
	}
	return header_string;
}

// Compiles a single source string into a Program wrapper
std::shared_ptr<Program> CompileFromSource(const std::string& source_string, const tart::DType& dtype,
										   const std::string& routine_name, const Device& device,
										   std::vector<std::string>& options,
										   const size_t run_preprocessor,
										   const bool silent)
{
	// Build preprocessor defines header
	auto header_string = BuildDefinesHeader(dtype, device);
	
	// Create a single-entry map for the kernel
	std::map<std::string, std::string> kernelSources;
	kernelSources[routine_name] = "#version 450\n\n" + header_string + source_string;
	
	// Create and return Program wrapper
	auto program = std::make_shared<Program>(device(), kernelSources);
	return program;
}

// Compiles multiple sources from a map
std::shared_ptr<Program> CompileFromSource(const std::string& source_string, const tart::DType& dtype,
		const std::string& routine_name, const Device& device,
		std::vector<std::string>& options,
		const size_t run_preprocessor,	// 0: platform dependent, 1: always, 2: never
		const bool silent, std::map<std::string, std::string>& kernelSources)
{
	auto header_string = BuildDefinesHeader(dtype, device);

	// For specific devices, use staggered/shuffled workgroup indices.
	if (device()->getMetadata().physicalDeviceProperties.vendorID == tart::VendorID::eAMD)
	{
		header_string += "#define USE_STAGGERED_INDICES 1\n";
	}
	
	// append the header to everything
	for (auto& pair : kernelSources)
	{
		pair.second = "#version 450\n\n" + header_string + pair.second;
	}
	
	auto program = std::make_shared<Program>(device(), kernelSources);
	return program;
}
// =================================================================================================
}	// namespace clblast
