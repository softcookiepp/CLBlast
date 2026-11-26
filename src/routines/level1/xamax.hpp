
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the functional interface for the Xamax routine.
//
// =================================================================================================

#ifndef CLBLAST_ROUTINES_XAMAX_H_
#define CLBLAST_ROUTINES_XAMAX_H_

#include <cstdint>
#include <cstddef>
#include <string>

#include "database/database.hpp"
#include "utilities/backend.hpp"

namespace clblast
{
// =================================================================================================

// Functional implementation of Xamax (index of absolute maximum)
void DoAmax(const tart::DType& dtype, const tart::device_ptr& device,
            const uint32_t n, const tart::buffer_ptr& imax_buffer, const size_t imax_offset,
            const tart::buffer_ptr& x_buffer, const size_t x_offset, const size_t x_inc,
            uint32_t max_flag, uint32_t min_flag, uint32_t amin_flag, const Databases& db);

// =================================================================================================
}	// namespace clblast

// CLBLAST_ROUTINES_XAMAX_H_
#endif
