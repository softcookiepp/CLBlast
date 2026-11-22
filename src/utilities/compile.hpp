
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the CLBlast way to compile a kernel from source, used for the library and for
// the auto-tuners.
//
// =================================================================================================

#ifndef CLBLAST_UTILITIES_COMPILE_H_
#define CLBLAST_UTILITIES_COMPILE_H_

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// Compiles a program from source code (Vulkan)
std::shared_ptr<Program> CompileFromSource(const std::string& source_string, const tart::DType& dtype,
                                           const std::string& routine_name, const Device& device,
                                           std::vector<std::string>& options,
                                           const size_t run_preprocessor,
                                           const bool silent = false);
// Multi-kernel version
std::shared_ptr<Program> CompileFromSource(const std::string& source_string, const tart::DType& dtype,
                                           const std::string& routine_name, const Device& device,
                                           std::vector<std::string>& options,
                                           const size_t run_preprocessor,
                                           const bool silent, std::map<std::string, std::string>& kernelSources);

// =================================================================================================
}  // namespace clblast

// CLBLAST_UTILITIES_COMPILE_H_
#endif
