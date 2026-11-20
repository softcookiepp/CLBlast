
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Tester class (see the header for information about the class).
//
// =================================================================================================

#include "test/correctness/tester.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "test/test_utilities.hpp"
#include "utilities/backend.hpp"
#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// Relative error margins
template <typename T>
float getRelativeErrorMargin() {
	return 0.005f;	// 0.5% is considered acceptable for float/double-precision
}
template float getRelativeErrorMargin<float>();		// as the above default
template float getRelativeErrorMargin<double>();	 // as the above default
template float getRelativeErrorMargin<float2>();	 // as the above default
template float getRelativeErrorMargin<double2>();	// as the above default
template <>
float getRelativeErrorMargin<half>() {
	return 0.080f;	// 8% (!) error is considered acceptable for half-precision
}

// Absolute error margins
template <typename T>
float getAbsoluteErrorMargin() {
	return 0.001f;
}
template float getAbsoluteErrorMargin<float>();		// as the above default
template float getAbsoluteErrorMargin<double>();	 // as the above default
template float getAbsoluteErrorMargin<float2>();	 // as the above default
template float getAbsoluteErrorMargin<double2>();	// as the above default
template <>
float getAbsoluteErrorMargin<half>() {
	return 0.15f;	// especially small values are inaccurate for half-precision
}

// L2 error margins
template <typename T>
double getL2ErrorMargin() {
	return 0.0f;	// zero means don't look at the L2 error margin at all, use the other metrics
}
template double getL2ErrorMargin<float>();		// as the above default
template double getL2ErrorMargin<double>();	 // as the above default
template double getL2ErrorMargin<float2>();	 // as the above default
template double getL2ErrorMargin<double2>();	// as the above default
template <>
double getL2ErrorMargin<half>() {
	return 0.05;	// half-precision results are considered OK as long as the L2 error is low enough
}

// Error margin: numbers beyond this value are considered equal to inf or NaN
template <typename T>
T getAlmostInfNumber() {
	return static_cast<T>(1e35);	// used for correctness testing of TRSV and TRSM routines
}

// =================================================================================================

// General constructor for all CLBlast testers. It prints out the test header to stdout and sets-up
// the clBLAS library for reference.
template <typename T, typename U>
Tester<T, U>::Tester(const std::vector<std::string>& arguments, const bool silent, const std::string& name,
										 const std::vector<std::string>& options)
		: help_("Options given/available:\n"),
			device_(Device(GetArgument(arguments, help_, kArgDevice,
																						ConvertArgument(std::getenv("CLBLAST_DEVICE"), size_t{0})))),
			queue_(Queue(device_)),
			full_test_(CheckArgument(arguments, help_, kArgFullTest)),
			verbose_(CheckArgument(arguments, help_, kArgVerbose)),
			error_log_{},
			num_passed_{0},
			num_skipped_{0},
			num_failed_{0},
			print_count_{0},
			tests_passed_{0},
			tests_skipped_{0},
			tests_failed_{0} {
	options_ = options;

// Determines which reference is the default
#if defined(CLBLAST_REF_CBLAS)
	auto default_cblas = 0;
#endif
