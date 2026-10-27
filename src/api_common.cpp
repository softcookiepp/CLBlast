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
#include "routines/routines.hpp"
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

template <typename Type>
void FillCacheForPrecision(Queue& queue) {
	try {
		// Runs all the level 1 set-up functions that support all precisions
		Xswap(tart::getDType<Type>(), queue, nullptr);
		Xscal(tart::getDType<Type>(), queue, nullptr);
		Xcopy(tart::getDType<Type>(), queue, nullptr);
		Xaxpy(tart::getDType<Type>(), queue, nullptr);
		Xdot(tart::getDType<Type>(), queue, nullptr);
		Xnrm2(tart::getDType<Type>(), queue, nullptr);
		Xasum(tart::getDType<Type>(), queue, nullptr);
		Xsum(tart::getDType<Type>(), queue, nullptr);
		Xamax(tart::getDType<Type>(), queue, nullptr);
		Xamin(tart::getDType<Type>(), queue, nullptr);
		Xmax(tart::getDType<Type>(), queue, nullptr);
		Xmin(tart::getDType<Type>(), queue, nullptr);

		// Runs all the level 2 set-up functions that support all precisions
		Xgemv(tart::getDType<Type>(), queue, nullptr);
		Xgbmv<Type>(queue, nullptr);
		Xtrmv<Type>(queue, nullptr);
		Xtbmv<Type>(queue, nullptr);
		Xtpmv<Type>(queue, nullptr);

		// Runs all the level 3 set-up functions that support all precisions
		Xgemm<Type>(queue, nullptr);
		Xsymm<Type>(queue, nullptr);
		Xsyrk<Type>(queue, nullptr);
		Xsyr2k<Type>(queue, nullptr);
		Xtrmm<Type>(queue, nullptr);

		// Runs all the non-BLAS set-up functions
		Xomatcopy<Type>(queue, nullptr);

	} catch (const RuntimeErrorCode& e) {
		if (e.status() != StatusCode::kNoDoublePrecision && e.status() != StatusCode::kNoHalfPrecision) {
			throw;
		}
	}
}

template <typename Real, typename Complex>
void FillCacheForPrecision(Queue& queue) {
	try {
		FillCacheForPrecision<Real>(queue);
		FillCacheForPrecision<Complex>(queue);

		// Runs all the level 1 set-up functions that don't support all precisions
		Xdotu(tart::getDType<Complex>(), queue, nullptr);
		Xdotc(tart::getDType<Complex>(), queue, nullptr);

		// Runs all the level 2 set-up functions that don't support all precisions
		Xhemv<Complex>(queue, nullptr);
		Xhbmv<Complex>(queue, nullptr);
		Xhpmv<Complex>(queue, nullptr);
		Xsymv<Real>(queue, nullptr);
		Xsbmv<Real>(queue, nullptr);
		Xspmv<Real>(queue, nullptr);
		Xtrmv<Real>(queue, nullptr);
		Xtrmv<Complex>(queue, nullptr);
		Xger<Real>(queue, nullptr);
		Xgeru<Complex>(queue, nullptr);
		Xgerc<Complex>(queue, nullptr);
		Xher<Complex, Real>(queue, nullptr);
		Xhpr<Complex, Real>(queue, nullptr);
		Xher2<Complex>(queue, nullptr);
		Xhpr2<Complex>(queue, nullptr);
		Xsyr<Real>(queue, nullptr);
		Xspr<Real>(queue, nullptr);
		Xsyr2<Real>(queue, nullptr);
		Xspr2<Real>(queue, nullptr);

		// Runs all the level 3 set-up functions that don't support all precisions
		Xhemm<Complex>(queue, nullptr);
		Xherk<Complex, Real>(queue, nullptr);
		Xher2k<Complex, Real>(queue, nullptr);
		Xspr2<Real>(queue, nullptr);

	} catch (const RuntimeErrorCode& e) {
		if (e.status() != StatusCode::kNoDoublePrecision && e.status() != StatusCode::kNoHalfPrecision) {
			throw;
		}
	}
}

// Fills the cache with all binaries for a specific device
StatusCode FillCache(const RawDeviceID device) {
	try {
		// Creates a sample context and queue to match the normal routine calling conventions
		auto device_cpp = Device(device);
		auto queue = Queue(device_cpp);

		FillCacheForPrecision<half>(queue);
		FillCacheForPrecision<float, float2>(queue);
		FillCacheForPrecision<double, double2>(queue);

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
		// Retrieves the device name
		const auto device_cpp = Device(device);
		const auto platform_id = 0;
		const auto device_name = GetDeviceName(device_cpp);

		// Retrieves the database values
		auto in_cache = false;
		auto database =
				DatabaseCache::Instance().Get(DatabaseKeyRef{device->getMetadata().deviceUUID, precisionToDataType(precision), kernel_name}, &in_cache);
		if (!in_cache) {
			log_debug("Searching database for kernel '" + kernel_name + "'");
			database = Database(device_cpp(), kernel_name, precisionToDataType(precision));
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
	#if 0
		throw std::runtime_error("needs reimplementation");
	#else
		try
		{
			tart::DType dt = precisionToDataType(precision);
			
			// Retrieves the device name
			const auto device_cpp = Device(device);
			const auto platform_id = 0;
			const auto device_name = GetDeviceName(device_cpp);

			// Retrieves the current database values to verify whether the new ones are complete
			auto in_cache = false;
			auto current_database =
					DatabaseCache::Instance().Get(DatabaseKeyRef{device->getMetadata().deviceUUID, dt, kernel_name}, &in_cache);
			if (!in_cache) {
				log_debug("Searching database for kernel '" + kernel_name + "'");
				current_database = Database(device_cpp(), kernel_name, dt);
			}

			// Verifies the parameters size
			const auto current_parameter_names = current_database.GetParameterNames();
			if (current_parameter_names.size() > parameters.size()) {
				return StatusCode::kMissingOverrideParameter;
			}

			// Retrieves the names and values separately and in the same order as the existing database
			auto parameter_values = database::Params{0};
			auto i = size_t{0};
			for (const auto& current_param : current_parameter_names) {
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
					database::NewDatabaseEntry{kernel_name, precisionToDataType(precision), current_parameter_names, current_parameter_names, {database_vendor}};
			const auto database_entries = std::vector<database::NewDatabaseEntry>{database_entry};
			const auto database = Database(device_cpp(), kernel_name, precisionToDataType(precision));

			// Removes the old database entry and stores the new one in the cache
			DatabaseCache::Instance().Remove(DatabaseKey{device->getMetadata().deviceUUID, dt, kernel_name});
			DatabaseCache::Instance().Store(DatabaseKey{device->getMetadata().deviceUUID, dt, kernel_name}, Database(database));

		} catch (...) {
			return DispatchException();
		}
	#endif
	return StatusCode::kSuccess;
}

// =================================================================================================
}	// namespace clblast
