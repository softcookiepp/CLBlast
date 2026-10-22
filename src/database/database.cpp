
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Database class (see the header for information about the class).
//
// =================================================================================================

#include "database/database.hpp"

#include <cstddef>
#include <list>
#include <memory>
#include <string>
#include <vector>

#include "database/database_structure.hpp"
#include "database/kernels/copy/copy.hpp"
#include "database/kernels/gemm_routine/gemm_routine.hpp"
#include "database/kernels/invert/invert.hpp"
#include "database/kernels/pad/pad.hpp"
#include "database/kernels/padtranspose/padtranspose.hpp"
#include "database/kernels/transpose/transpose.hpp"
#include "database/kernels/trsv_routine/trsv_routine.hpp"
#include "database/kernels/xaxpy/xaxpy.hpp"
#include "database/kernels/xconvgemm/xconvgemm.hpp"
#include "database/kernels/xdot/xdot.hpp"
#include "database/kernels/xgemm/xgemm.hpp"
#include "database/kernels/xgemm_direct/xgemm_direct.hpp"
#include "database/kernels/xgemv/xgemv.hpp"
#include "database/kernels/xgemv_fast/xgemv_fast.hpp"
#include "database/kernels/xgemv_fast_rot/xgemv_fast_rot.hpp"
#include "database/kernels/xger/xger.hpp"
#include "utilities/backend.hpp"
#include "utilities/clblast_exceptions.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

std::vector<database::NewDatabaseEntry> Database::newDatabase = {};

// =================================================================================================

// Constructor, computing device properties and populating the parameter-vector from the database.
// This takes an optional overlay database in case of custom tuning or custom kernels.
Database::Database(const tart::device_ptr& device, const std::string& kernel_name, const tart::DType& dt)
	: parameters_(std::make_shared<database::Parameters>())
{
	// Initializes the static variable on first use. At this point we are sure all global variables are initialized
	if (newDatabase.size() == 0)
	{
		newDatabase = std::vector<database::NewDatabaseEntry>{
			database::NewDbXaxpyHalf,
			database::NewDbXaxpySingle,
			database::NewDbXaxpyDouble,
			database::NewDbXaxpyComplexSingle,
			database::NewDbXaxpyComplexDouble,
			database::NewDbXdotHalf,
			database::NewDbXdotSingle,
			database::NewDbXdotDouble,
			database::NewDbXdotComplexSingle,
			database::NewDbXdotComplexDouble,
			database::NewDbXgemvHalf,
			database::NewDbXgemvSingle,
			database::NewDbXgemvDouble,
			database::NewDbXgemvComplexSingle,
			database::NewDbXgemvComplexDouble,
			database::NewDbXgemvFastHalf,
			database::NewDbXgemvFastSingle,
			database::NewDbXgemvFastDouble,
			database::NewDbXgemvFastComplexSingle,
			database::NewDbXgemvFastComplexDouble,
			database::NewDbXgemvFastRotHalf,
			database::NewDbXgemvFastRotSingle,
			database::NewDbXgemvFastRotDouble,
			database::NewDbXgemvFastRotComplexSingle,
			database::NewDbXgemvFastRotComplexDouble,
			database::NewDbXgerHalf,
			database::NewDbXgerSingle,
			database::NewDbXgerDouble,
			database::NewDbXgerComplexSingle,
			database::NewDbXgerComplexDouble,
			database::NewDbXgemmHalf,
			database::NewDbXgemmSingle,
			database::NewDbXgemmDouble,
			database::NewDbXgemmComplexSingle,
			database::NewDbXgemmComplexDouble,
			database::NewDbXgemmDirectHalf,
			database::NewDbXgemmDirectSingle,
			database::NewDbXgemmDirectDouble,
			database::NewDbXgemmDirectComplexSingle,
			database::NewDbXgemmDirectComplexDouble,
			database::NewDbXconvgemmHalf,
			database::NewDbXconvgemmSingle,
			database::NewDbXconvgemmDouble,
			database::NewDbXconvgemmComplexSingle,
			database::NewDbXconvgemmComplexDouble,
			database::NewDbCopyHalf,
			database::NewDbCopySingle,
			database::NewDbCopyDouble,
			database::NewDbCopyComplexSingle,
			database::NewDbCopyComplexDouble,
			database::NewDbPadHalf,
			database::NewDbPadSingle,
			database::NewDbPadDouble,
			database::NewDbPadComplexSingle,
			database::NewDbPadComplexDouble,
			database::NewDbTransposeHalf,
			database::NewDbTransposeSingle,
			database::NewDbTransposeDouble,
			database::NewDbTransposeComplexSingle,
			database::NewDbTransposeComplexDouble,
			database::NewDbPadtransposeHalf,
			database::NewDbPadtransposeSingle,
			database::NewDbPadtransposeDouble,
			database::NewDbPadtransposeComplexSingle,
			database::NewDbPadtransposeComplexDouble,
			database::NewDbInvertHalf,
			database::NewDbInvertSingle,
			database::NewDbInvertDouble,
			database::NewDbInvertComplexSingle,
			database::NewDbInvertComplexDouble,
			database::NewDbGemmRoutineHalf,
			database::NewDbGemmRoutineSingle,
			database::NewDbGemmRoutineDouble,
			database::NewDbGemmRoutineComplexSingle,
			database::NewDbGemmRoutineComplexDouble,
			database::NewDbTrsvRoutineHalf,
			database::NewDbTrsvRoutineSingle,
			database::NewDbTrsvRoutineDouble,
			database::NewDbTrsvRoutineComplexSingle,
			database::NewDbTrsvRoutineComplexDouble
		};
	}
	
	// search through new database implementation if possible
	auto search_result = database::Parameters();
	
	uint32_t vendorID = device->getMetadata().physicalDeviceProperties.vendorID;
	uint32_t deviceID = device->getMetadata().physicalDeviceProperties.deviceID;
	search_result = Search(kernel_name, vendorID, deviceID, dt, {newDatabase});
	if (search_result.size() != 0)
	{
		parameters_->insert(search_result.begin(), search_result.end());
	}
	
	if (search_result.size() == 0) {
		throw RuntimeErrorCode(StatusCode::kDatabaseError);
	}
}

// =================================================================================================

// Returns a list of OpenCL pre-processor defines in string form
std::string Database::GetDefines() const {
	std::string defines{};
	for (auto& parameter : mDefs)
	{
		defines += "#define " + parameter.first + " " + ToString(static_cast<size_t>(parameter.second)) + "\n";
	}
	return defines;
}

// ... or just the values as string
std::string Database::GetValuesString() const {
	std::string defines{};
	for (auto& parameter : *parameters_) {
		defines += "_" + ToString(parameter.second);
	}
	return defines;
}

// Retrieves the names of all the parameters
std::vector<std::string> Database::GetParameterNames() const {
	auto parameter_names = std::vector<std::string>();
	for (auto& parameter : *parameters_) {
		parameter_names.push_back(parameter.first);
	}
	return parameter_names;
}

// =================================================================================================

// Searches a particular database for the right kernel and precision
database::Parameters Database::Search(
	const std::string& this_kernel, const uint32_t vendorID,
	const uint32_t deviceID, const tart::DType& dtype,
	const std::vector<database::NewDatabaseEntry>& db)
{
	// Selects the right kernel
	for (auto& db : db)
	{
		if (db.kernel == this_kernel && db.precision == dtype)
		{
			// Searches for the right vendor and device type, or selects the default if unavailable
			return SearchVendorAndType(vendorID, deviceID, db.vendors, db.def_names, db.spec_names);
		}
	}
	throw std::runtime_error("meep!");

	// If we reached this point, the entry was not found in this database
	return database::Parameters();
}

database::Parameters Database::SearchVendorAndType(const uint32_t targetVendorID,
		const uint32_t targetDeviceID,
		const std::vector<database::NewDatabaseVendor>& vendors,
		const std::vector<std::string>& def_names,
		const std::vector<std::string>& spec_names)
{
	database::Parameters params;
	for (auto& vendor : vendors)
	{
		if (vendor.vendorID == targetVendorID || vendor.vendorID == database::kDefaultVendorID)
		{
			log_debug(std::string("Found architectures of vendor "));

			// Searches the architecture; if unavailable returns the vendor's default parameters
			params = SearchDevice(targetDeviceID, vendor.devices, def_names, spec_names);
			if (params.size() > 0) return params;
		}
	}
	throw std::runtime_error("meep!");
	return database::Parameters();
}

database::Parameters Database::SearchDevice(
	const uint32_t targetDeviceID,
	const std::vector<database::NewDatabaseDevice>& devices,
	const std::vector<std::string>& def_names,
	const std::vector<std::string>& spec_names)
{
	for (auto& device : devices)
	{
		if (device.deviceID == targetDeviceID || device.deviceID == database::kDefaultDeviceID)
		{
			log_debug("Found parameters for device");

			// Sets the parameters accordingly
			auto parameters = database::Parameters();
			if (spec_names.size() > device.specs.size() || def_names.size() > device.defs.size())
			{
				// this should not happen
				throw std::invalid_argument("parameter size mismatch");
				return database::Parameters();
			}
			// this will be changed later.
			for (size_t i = 0; i < def_names.size(); i += 1)
			{
				parameters[def_names[i]] = static_cast<size_t>(device.defs[i]);
				mDefs[def_names[i]] = static_cast<uint32_t>(device.defs[i]);
			}
			for (size_t i = 0; i < spec_names.size(); i += 1)
			{
				parameters[spec_names[i]] = static_cast<size_t>(device.specs[i]);
				mSpecs[spec_names[i]] = static_cast<uint32_t>(device.specs[i]);
			}
			return parameters;
		}
	}
	throw std::runtime_error("meep!");
	return database::Parameters();
}

// Helper to convert from database format to proper types
std::string Database::CharArrayToString(const database::Name char_array) const {
	auto result = std::string(char_array.data());
	result.erase(result.find_last_not_of(" \t\n\r\f\v") + 1);
	return result;
}

// =================================================================================================
}	// namespace clblast
