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

#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = WGS, local_size_y = 1, local_size_z = 1) in;
#endif

#if USE_BDA == 0
	layout(binding = 0, std430) buffer xgm_buf { realV xgm[]; };
	layout(binding = 1, std430) buffer ygm_buf { realV ygm[]; };
	layout(binding = 2, std430) buffer zgm_buf { realV zgm[]; };
#endif

layout(push_constant) uniform XhadFaster
{
	int n; real_arg arg_alpha; real_arg arg_beta;
#if USE_BDA
	__global real* restrict xgm;
	__global real* restrict ygm;
	__global real* zgm;
#endif
};

// Faster version of the kernel without offsets and strided accesses. Also assumes that 'n' is
// dividable by 'VW', 'WGS' and 'WPT'.

void main()
{
	if (!(n % VW == 0 && n % WPT == 0 && n % WGS == 0) ) return;
	
	const real alpha = GetRealArg(arg_alpha);
	const real beta = GetRealArg(arg_beta);

	//#pragma unroll
	for (int _w = 0; _w < WPT; _w += 1) {
		const int id = _w*get_global_size(0) + get_global_id(0);
		realV xvalue = xgm[id];
		realV yvalue = ygm[id];
		realV zvalue = zgm[id];
		realV result;
		realV alpha_times_x;
		alpha_times_x = MultiplyVector(alpha_times_x, alpha, xvalue);
		result = MultiplyVectorVector(result, alpha_times_x, yvalue);
		zgm[id] = MultiplyAddVector(result, beta, zvalue);
	}
}

// =================================================================================================

// End of the C++11 raw string literal
//)"

// =================================================================================================
