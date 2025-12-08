#version 450
#include "../common.glsl"
#include "level1.glsl"
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

// =================================================================================================

// Full version of the kernel with offsets and strided accesses
#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = WGS, local_size_y = 1, local_size_z = 1) in;
#endif

#if USE_BDA == 0
	layout(binding = 0, std430) buffer xgm_buf { real xgm[]; };
#endif

layout(push_constant) uniform Xscal
{
	int n;
	real_arg arg_alpha;
#if USE_BDA
	__global real* xgm;
#endif
	int x_offset;
	int x_inc;
} args;

// Xscal
void main()
{
	const real alpha = GetRealArg(args.arg_alpha);

	// Loops over the work that needs to be done (allows for an arbitrary number of threads)
	for (int id = get_global_id(0); id<args.n; id += get_global_size(0))
	{
		real xvalue = INDEX(xgm, id*args.x_inc + args.x_offset);
		real result;
		Multiply(result, alpha, xvalue);
		INDEX(xgm, id*args.x_inc + args.x_offset) = result;
	}
}

// =================================================================================================

// End of the C++11 raw string literal
//)"

// =================================================================================================
