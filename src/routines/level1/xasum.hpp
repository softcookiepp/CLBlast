
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the functional interface for the Xasum routine.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XASUM_H_
#define CLBLAST_ROUTINES_XASUM_H_

#include <cstdint>
#include <cstddef>
#include <string>

#include "database/database.hpp"
#include "utilities/backend.hpp"

namespace clblast {
// =================================================================================================

// Functional implementation of Xasum (absolute sum)
void DoAsum(const tart::DType& dtype, const tart::device_ptr& device,
            const size_t n, const tart::buffer_ptr& asum_buffer, const size_t asum_offset,
            const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
            uint32_t sum_flag, const Databases& db);

// =================================================================================================
}	// namespace clblast

// CLBLAST_ROUTINES_XASUM_H_
#endif
