#version 450
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xscal kernel. It contains one fast vectorized version in case of unit
// strides (incx=1) and no offsets (offx=0). Another version is more general, but doesn't support
// vector data-types.
//
// This kernel uses the level-1 BLAS common tuning parameters.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
//R"(
#include "../common.glsl"
#include "level1.glsl"
// =================================================================================================

// Faster version of the kernel without offsets and strided accesses. Also assumes that 'n' is
// dividable by 'VW', 'WGS' and 'WPT'.

#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = WGS, local_size_y = 1, local_size_z = 1) in;
#endif

#if USE_BDA == 0
	layout(binding = 0, std430) buffer xgm_buf { realV xgm[]; };
#endif

layout(push_constant, std430) uniform XscalFast
{
	int n;
	real_arg arg_alpha;
#if USE_BDA
	realV_ptr_t xgm;
#endif
};

// XscalFast
void main()
{
	const real alpha = GetRealArg(arg_alpha);
	for (int _w = 0; _w < WPT; _w += 1)
	{
		const int id = _w*get_global_size(0) + get_global_id(0);
		realV xvalue = indexGM(xgm, id);
		realV result;
		result = MultiplyVector(result, alpha, xvalue);
		indexGM(xgm, id) = result;
	}
}

// =================================================================================================

// End of the C++11 raw string literal
//)"

// =================================================================================================
