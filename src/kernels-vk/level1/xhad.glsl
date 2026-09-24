#version 450

// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xhad kernel. It contains one fast vectorized version in case of unit
// strides (incx=incy=incz=1) and no offsets (offx=offy=offz=0). Another version is more general,
// but doesn't support vector data-types. Based on the XAXPY kernels.
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

// A vector-vector multiply function. See also level1.opencl for a vector-scalar version
realV MultiplyVectorVector(realV cvec, const realV aval, const realV bvec) {
	#if VW == 1
		Multiply(cvec, aval, bvec);
	#else
		vMultiply(cvec, aval, bvec, VW);
	#endif
	return cvec;
}

// =================================================================================================

// Full version of the kernel with offsets and strided accesses
layout(local_size_x = WGS, local_size_y = 1, local_size_z = 1) in;

#if USE_BDA == 0
	layout(binding = 0, std430) buffer xgm_buf { real xgm[]; };
	layout(binding = 1, std430) buffer ygm_buf { real ygm[]; };
	layout(binding = 2, std430) buffer zgm_buf { real zgm[]; };
#endif

layout(push_constant) uniform Xhad
{
	int n; real_arg arg_alpha; real_arg arg_beta;
	#if USE_BDA
		__global real* restrict xgm;
	#endif
	int x_offset; int x_inc;
	#if USE_BDA
		__global real* restrict ygm;
	#endif
	int y_offset; int y_inc;
	#if USE_BDA
		__global real* zgm;
	#endif
	int z_offset; int z_inc;
};

void main()
{
	const real alpha = GetRealArg(arg_alpha);
	const real beta = GetRealArg(arg_beta);
	
	// Loops over the work that needs to be done (allows for an arbitrary number of threads)
	for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
		real xvalue = xgm[id*x_inc + x_offset];
		real yvalue = ygm[id*y_inc + y_offset];
		real zvalue = zgm[id*z_inc + z_offset];
		real result;
		real alpha_times_x;
		Multiply(alpha_times_x, alpha, xvalue);
		Multiply(result, alpha_times_x, yvalue);
		MultiplyAdd(result, beta, zvalue);
		zgm[id*z_inc + z_offset] = result;
	}
}
// =================================================================================================

// End of the C++11 raw string literal
//)"

// =================================================================================================
