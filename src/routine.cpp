
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Routine base class (see the header for information about the class).
//
// =================================================================================================

#include "routine.hpp"

#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "database/database_structure.hpp"
#include "utilities/backend.hpp"
#include "utilities/clblast_exceptions.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
	
bool Routine::sCreatedDatabase = false;
DatabaseKey Routine::sFirstDatabaseKey;
// =================================================================================================

// For each kernel this map contains a list of routines it is used in
const std::vector<std::string> Routine::routines_axpy = {"AXPY", "COPY", "SCAL", "SWAP"};
const std::vector<std::string> Routine::routines_dot = {"AMAX", "ASUM", "DOT",	"DOTC", "DOTU",
																												"MAX",	"MIN",	"NRM2", "SUM"};
const std::vector<std::string> Routine::routines_ger = {"GER",	"GERC", "GERU", "HER", "HER2", "HPR",
																												"HPR2", "SPR",	"SPR2", "SYR", "SYR2"};
const std::vector<std::string> Routine::routines_gemv = {"GBMV", "GEMV", "HBMV", "HEMV", "HPMV", "SBMV",
																												 "SPMV", "SYMV", "TMBV", "TPMV", "TRMV", "TRSV"};
const std::vector<std::string> Routine::routines_gemm = {"GEMM", "HEMM", "SYMM", "TRMM"};
const std::vector<std::string> Routine::routines_gemm_syrk = {"GEMM",	"HEMM", "HER2K", "HERK", "SYMM",
																															"SYR2K", "SYRK", "TRMM",	"TRSM"};
const std::vector<std::string> Routine::routines_trsm = {"TRSM"};
const std::unordered_map<std::string, const std::vector<std::string>> Routine::routines_by_kernel = {
		{"Xaxpy", routines_axpy},
		{"Xdot", routines_dot},
		{"Xgemv", routines_gemv},
		{"XgemvFast", routines_gemv},
		{"XgemvFastRot", routines_gemv},
		{"Xtrsv", routines_gemv},
		{"Xger", routines_ger},
		{"Copy", routines_gemm_syrk},
		{"Pad", routines_gemm_syrk},
		{"Transpose", routines_gemm_syrk},
		{"Padtranspose", routines_gemm_syrk},
		{"Xgemm", routines_gemm_syrk},
		{"XgemmDirect", routines_gemm},
		{"GemmRoutine", routines_gemm},
		{"Invert", routines_trsm},
};
// =================================================================================================

// The constructor does all heavy work, errors are returned as exceptions
Routine::Routine(Queue& queue, EventPointer event, const std::string& name,
	const std::vector<std::string>& kernel_names, const tart::DType& dtype,
	const std::vector<database::NewDatabaseEntry>& userDatabase)
		: 
			mDtype(dtype),
			routine_name_(name),
			kernel_names_(kernel_names),
			queue_(queue),
			event_(event),
			device_(queue.GetDevice()),
			db_(kernel_names),
			mDevice(queue())
{
	InitDatabase(device_, kernel_names, dtype, userDatabase, db_);
}

void Routine::InitDatabase(const Device& device, const std::vector<std::string>& kernel_names,
	const tart::DType& precision, const std::vector<database::NewDatabaseEntry>& userDatabase, Databases& db)
{
	const auto platform_id = 0;
	for (const auto& kernel_name : kernel_names)
	{
		// Queries the cache to see whether or not the kernel parameter database is already there
		bool has_db;
		db(kernel_name) =
				DatabaseCache::Instance().Get(DatabaseKeyRef{device()->getMetadata().deviceUUID, precision, kernel_name}, &has_db);
		if (has_db) {
			continue;
		}
		DatabaseKey thisKey = {device()->getMetadata().deviceUUID, precision, kernel_name};
		
		if (sCreatedDatabase && thisKey == sFirstDatabaseKey)
			throw std::runtime_error("Should have created the database by now, but it was not found!");
		sCreatedDatabase = true;
		sFirstDatabaseKey = thisKey;

		// Builds the parameter database for this device and routine set and stores it in the cache
		log_debug("Searching database for kernel '" + kernel_name + "'");
		db(kernel_name) = Database(device(), kernel_name, precision);//, userDatabase);
		DatabaseCache::Instance().Store(DatabaseKey{device()->getMetadata().deviceUUID, precision, kernel_name},
																		Database{db(kernel_name)});
	}
}


// =================================================================================================
}	// namespace clblast
