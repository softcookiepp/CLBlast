// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the common (non-OpenCL-specific) functions of the CLBlast API.
//
// =================================================================================================

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "cache.hpp"
#include "database/database_structure.hpp"
#include "utilities/backend.hpp"
#include "utilities/clblast_exceptions.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// Clears the cache of stored binaries
StatusCode ClearCache() {
	try {
		ProgramCache::Instance().Invalidate();
	} catch (...) {
		return DispatchException();
	}
	return StatusCode::kSuccess;
}

// =================================================================================================

// Retrieves the current tuning parameters for this device-precision-kernel combination
StatusCode RetrieveParameters(const RawDeviceID device, const std::string& kernel_name, const Precision precision,
															std::unordered_map<std::string, size_t>& parameters) {
	try {
		// Retrieves the database values
		auto in_cache = false;
		auto database =
				DatabaseCache::Instance().Get(DatabaseKeyRef{device->getMetadata().deviceUUID, precisionToDataType(precision), kernel_name}, &in_cache);
		if (!in_cache) {
			log_debug("Searching database for kernel '" + kernel_name + "'");
			database = Database(device(), kernel_name, precisionToDataType(precision));
		}

		// Retrieves the parameters
		for (const auto& p : database.getDefs())
			parameters[p.first] = static_cast<size_t>(p.second);
		for (const auto& p : database.getSpecs())
			parameters[p.first] = static_cast<size_t>(p.second);
	} catch (...) {
		return DispatchException();
	}
	return StatusCode::kSuccess;
}

// Overrides the tuning parameters for this device-precision-kernel combination
StatusCode OverrideParameters(const RawDeviceID device, const std::string& kernel_name, const Precision precision,
															const std::unordered_map<std::string, size_t>& parameters)
{
	try
	{
		tart::DType dt = precisionToDataType(precision);

		// Retrieves the current database values to verify whether the new ones are complete
		auto in_cache = false;
		auto current_database =
				DatabaseCache::Instance().Get(DatabaseKeyRef{device->getMetadata().deviceUUID, dt, kernel_name}, &in_cache);
		if (!in_cache) {
			log_debug("Searching database for kernel '" + kernel_name + "'");
			current_database = Database(device(), kernel_name, dt);
		}

		// Verifies the parameters size
		const auto all_params = [&]() {
			auto params = std::vector<std::string>();
			for (const auto& p : current_database.getDefs())
				params.push_back(p.first);
			for (const auto& p : current_database.getSpecs())
				params.push_back(p.first);
			return params;
		}();
		if (all_params.size() > parameters.size()) {
			return StatusCode::kMissingOverrideParameter;
		}

		// Retrieves the names and values separately and in the same order as the existing database
		auto parameter_values = database::Params{0};
		auto i = size_t{0};
		for (const auto& current_param : all_params) {
			if (parameters.find(current_param) == parameters.end()) {
				return StatusCode::kMissingOverrideParameter;
			}
			const auto parameter_value = parameters.at(current_param);
			parameter_values[i] = parameter_value;
			++i;
		}

		// Creates a small custom database based on the provided parameters
		const auto database_device = database::NewDatabaseDevice{database::kDeviceNameDefault, database::kDefaultDeviceID, parameter_values, parameter_values};

		const auto database_vendor = database::NewDatabaseVendor{database::kDefaultVendorID, {database_device}};
		const auto database_entry =
				database::NewDatabaseEntry{kernel_name, precisionToDataType(precision), all_params, all_params, {database_vendor}};
		const auto database = std::vector<database::NewDatabaseEntry>{database_entry};
		const auto db = Database(device(), kernel_name, precisionToDataType(precision));

		// Removes the old database entry and stores the new one in the cache
		DatabaseCache::Instance().Remove(DatabaseKey{device->getMetadata().deviceUUID, dt, kernel_name});
		DatabaseCache::Instance().Store(DatabaseKey{device->getMetadata().deviceUUID, dt, kernel_name}, Database(db));

	} catch (...) {
		return DispatchException();
	}
	return StatusCode::kSuccess;
}

// =================================================================================================
}	// namespace clblast
