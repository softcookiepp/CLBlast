
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>, Ekansh Jain
//
// This file decides which backend to use for the library (OpenCL or CUDA)
//
// =================================================================================================

#ifndef CLBLAST_BACKEND_HPP_
#define CLBLAST_BACKEND_HPP_

#ifdef OPENCL_API
#include <clblast.h>  // IWYU pragma: export

#include "clpp11.hpp"  // IWYU pragma: export

#elif VULKAN_API
#include <clblast_vk.h>
#include "vkpp11.hpp"

#endif

#include "cxpp11_common.hpp"  // IWYU pragma: export

#endif
