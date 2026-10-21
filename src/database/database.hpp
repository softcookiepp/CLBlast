
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Database class, providing a static variable holding the actual database
// information. The class also provides utility functions to search the database and to access a
// found entry by parameter-key. The database itself is filled in the corresponding source-file and
// partially also by the database/xxxxx.h files, in which kernel-specific parameters are found.
//
// =================================================================================================

#ifndef CLBLAST_DATABASE_H_
#define CLBLAST_DATABASE_H_

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "database/database_structure.hpp"
#include "utilities/backend.hpp"
#include "utilities/clblast_exceptions.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
class Database {
 public:
	// The OpenCL device vendors
	static const std::string kDeviceVendorAll;

	// The database consists of separate database entries, stored together in a vector

	static std::vector<database::NewDatabaseEntry> newDatabase;

	Database() = default;

	// The constructor with a user-provided database overlay (potentially an empty vector)
	explicit Database(const tart::device_ptr& device, const std::string& kernel_name, const Precision precision,
										const std::vector<database::NewDatabaseEntry>& overlay);

	// Accessor of values by key
	size_t operator[](const std::string& key) const { return parameters_->find(key)->second; }
	bool exists(const std::string& key) const { return (parameters_->count(key) == 1); }

	// Obtain a list of OpenCL pre-processor defines based on the parameters
	std::string GetDefines() const;

	// Retrieves the values or names of all the parameters
	std::string GetValuesString() const;
	std::vector<std::string> GetParameterNames() const;
	const database::Parameters& GetParameters() const { return *parameters_; }

 private:
	// Search method functions, returning a set of parameters (possibly empty)
	database::Parameters Search(const std::string& this_kernel, const uint32_t vendorID,
		const uint32_t deviceID,
		const tart::DType& dtype,
		const std::vector<database::NewDatabaseEntry>& db) const;
	
	database::Parameters SearchDevice(
		const uint32_t targetDeviceID,
		const std::vector<database::NewDatabaseDevice>& devices,
		const std::vector<std::string>& parameter_names) const;
		
	database::Parameters SearchVendorAndType(
		const uint32_t targetVendorID, const uint32_t deviceID,
		const std::vector<database::NewDatabaseVendor>& vendors,
		const std::vector<std::string>& parameter_names) const;

	// Helper to convert from database format to proper types
	std::string CharArrayToString(const database::Name char_array) const;

	// Found parameters suitable for this device/kernel
	std::shared_ptr<database::Parameters> parameters_;
};

// =================================================================================================

// Multiple databases together in a map
class Databases {
 public:
	explicit Databases(const std::vector<std::string>& kernel_names) : kernel_names_(kernel_names) {}

	// Database accessor
	Database& operator()(const std::string& kernel_name) { return databases_[kernel_name]; }

	// Retrieves a parameter from the database
	size_t operator[](const std::string& key) const {
		for (const auto& kernel_name : kernel_names_) {
			const auto& kernel_db = databases_.find(kernel_name)->second;
			if (kernel_db.exists(key)) {
				return kernel_db[key];
			}
		}
		throw RuntimeErrorCode(StatusCode::kDatabaseError);
	}

 private:
	const std::vector<std::string> kernel_names_;
	std::unordered_map<std::string, Database> databases_;
};

// =================================================================================================
}	// namespace clblast

// CLBLAST_DATABASE_H_
#endif
