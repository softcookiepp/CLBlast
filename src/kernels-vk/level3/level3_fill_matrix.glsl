#version 450

// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the common functions and parameters specific for level 3 BLAS kernels.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
//R"(
#include "../common.glsl"

// =================================================================================================

#if USE_BDA == 0
	layout(binding = 0, std430) writeonly buffer dest_buf { real dest[]; };
#endif

// TODO: figure out what the local size should be

layout(push_constant) uniform FillMatrix
{
	int m; int n; int ld; int offset;
#if USE_BDA
	__global real* restrict dest;
#endif
	real_arg arg_value;
};
void main()
{
	const real value = (arg_value);
	const int id_one = get_global_id(0);
	const int id_two = get_global_id(1);
	if (id_one < m && id_two < n) {
		dest[id_two*ld + id_one + offset] = value;
	}
}


// =================================================================================================

// End of the C++11 raw string literal
//)"

// =================================================================================================
