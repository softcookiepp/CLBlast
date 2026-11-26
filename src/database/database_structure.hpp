
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file describes the database storage structures.
//
// =================================================================================================

#ifndef CLBLAST_DATABASE_DATABASE_STRUCTURE_H_
#define CLBLAST_DATABASE_DATABASE_STRUCTURE_H_

#include <array>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

// Include Vulkan BLAS header for Precision enum (only backend supported)
#include "clblast_vk.h"

namespace clblast {
// A special namespace to hold all the global constant variables (including the database entries)
namespace database {

// =================================================================================================

// Type alias for the database storage (arrays for fast compilation/efficiency)
using Name = std::array<char, 51>;			// name as stored in database (50 chars + string terminator)
using Params = std::vector<uint32_t>;	// parameters as stored in database

// Type alias after extracting from the database (sorted map for improved code readability)
using Parameters = std::map<std::string, uint32_t>;	// parameters after reading from DB

// default name
const Name kDeviceNameDefault = {"default"};

// Vulkan has no way of determining architecture that I am aware of.
// There are also efficiencies to be gained by not using any strings for searching through nested arrays
const uint32_t kDefaultDeviceID = 0;
const uint32_t kDefaultVendorID = 0;

struct NewDatabaseDevice
{
	Name name;
	uint32_t deviceID;
	Params parameters;	// parameter values
	Params defs; // define-based parameters
	Params specs; // specialization constants
};

struct NewDatabaseVendor
{
	uint32_t vendorID;
	std::vector<NewDatabaseDevice> devices;
};

struct NewDatabaseEntry
{
	std::string kernel;
	tart::DType precision;
	std::vector<std::string> def_names;
	std::vector<std::string> spec_names;
	std::vector<NewDatabaseVendor> vendors;
};

// =================================================================================================
}	// namespace database
}	// namespace clblast

// CLBLAST_DATABASE_DATABASE_STRUCTURE_H_
#endif
