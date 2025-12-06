#version 450
#include "../common.glsl"
#include "level1.glsl"
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xaxpy kernel. It contains one fast vectorized version in case of unit
// strides (incx=incy=1) and no offsets (offx=offy=0). Another version is more general, but doesn't
// support vector data-types. The general version has a batched implementation as well.
//
// This kernel uses the level-1 BLAS common tuning parameters.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
//R"(

// =================================================================================================

// Faster version of the kernel without offsets and strided accesses but with if-statement. Also
// assumes that 'n' is dividable by 'VW' and 'WPT'.
#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = WGS, local_size_y = 1, local_size_z = 1) in;
#endif

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer xgm_buf { realV xgm[]; };
	layout(binding = 1, std430) buffer ygm_buf { realV ygm[]; };
#endif

layout(push_constant) uniform XaxpyFaster
{
	int n;
	real_arg arg_alpha;
#if USE_BDA
	const __global realV* restrict xgm,
	__global realV* ygm
#endif
} args;

void main()
{
	const real alpha = GetRealArg(args.arg_alpha);

	const int num_usefull_threads = args.n / (VW * WPT);
	if (get_global_id(0) < num_usefull_threads) {
		//#pragma unroll
		for (int _w = 0; _w < WPT; _w += 1) {
			const int id = _w*num_usefull_threads + get_global_id(0);
			realV xvalue = xgm[id];
			realV yvalue = ygm[id];
			ygm[id] = MultiplyAddVector(yvalue, alpha, xvalue);
		}
	}
}

// =================================================================================================

// End of the C++11 raw string literal
//)"

// =================================================================================================
