
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements all the basic functionality for the BLAS routines. This class serves as a
// base class for the actual routines (e.g. Xaxpy, Xgemm). It contains common functionality such as
// compiling the OpenCL kernel, connecting to the database, etc.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINE_H_
#define CLBLAST_ROUTINE_H_

#include <initializer_list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "cache.hpp"
#include "database/database.hpp"
#include "database/database_structure.hpp"
#include "utilities/backend.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
class Routine {
 public:
	// Initializes db_, fetching cached database or building one
	static void InitDatabase(const Device& device, const std::vector<std::string>& kernel_names,
													 const Precision precision, const std::vector<database::DatabaseEntry>& userDatabase,
													 Databases& db);

	// Base class constructor. The user database is an optional extra database to override the
	// built-in database.
	// All heavy preparation work is done inside this constructor.
	// NOTE: the caller must provide the same userDatabase for each combination of device, precision
	// and routine list, otherwise the caching logic will break.
	explicit Routine(Queue& queue, EventPointer event, const std::string& name, const std::vector<std::string>& routines,
									 const Precision precision, const std::vector<database::DatabaseEntry>& userDatabase,
									 std::initializer_list<const char*> source
#if VULKAN_API
					, std::vector<std::string> entryPointNames = {}
#endif
									 );

	// List of kernel-routine look-ups
	static const std::vector<std::string> routines_axpy;
	static const std::vector<std::string> routines_dot;
	static const std::vector<std::string> routines_ger;
	static const std::vector<std::string> routines_gemv;
	static const std::vector<std::string> routines_gemm;
	static const std::vector<std::string> routines_gemm_syrk;
	static const std::vector<std::string> routines_trsm;
	static const std::unordered_map<std::string, const std::vector<std::string>> routines_by_kernel;

 private:
	// Initializes program_, fetching cached program or building one
	void InitProgram(std::initializer_list<const char*> source);

 protected:
	// Non-static variable for the precision
	const Precision precision_;

	// The routine's name and the corresponding kernels
	const std::string routine_name_;
	const std::vector<std::string> kernel_names_;

	// The OpenCL objects, accessible only from derived classes
	Queue queue_;
	EventPointer event_;
	const Context context_;
	const Device device_;

	// Compiled program (either retrieved from cache or compiled in slow path)
	std::shared_ptr<Program> program_;

	// Connection to the database for all the device-specific parameters
	Databases db_;
	
#if VULKAN_API
	std::vector<std::string> mEntryPointNames;
	
	// submit if sequence was not provided; otherwise, do nothing
	tart::command_sequence_ptr getWorkingSequence(const tart::command_sequence_ptr& sequence);
	void submitIfNeeded(const tart::command_sequence_ptr& sequence, const tart::command_sequence_ptr& workingSequence,
		const std::vector<Event>& waitForEvents, const tart::event_ptr& signalEvent);
#endif
};

// =================================================================================================
}	// namespace clblast

// CLBLAST_ROUTINE_H_
#endif
