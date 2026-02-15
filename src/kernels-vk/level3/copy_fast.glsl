#version 450

// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the common kernels shared among different BLAS routines. This file contains
// kernels to copy matrices.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
//R"(
#include "../common.glsl"
#include "level3.glsl"
// =================================================================================================

// Data-widths
#if COPY_VW == 1
	#define realC real
#elif COPY_VW == 2
	#define realC real2
#elif COPY_VW == 4
	#define realC real4
#elif COPY_VW == 8
	#define realC real8
#elif COPY_VW == 16
	#define realC real16
#endif

// =================================================================================================

// Fast copy kernel. Requires 'ld' and the number of threads in dimension 0 to be a multiple of
// COPY_VW. Also requires both matrices to be of the same dimensions and without offset.
#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = COPY_DIMX, local_size_y = COPY_DIMY, local_size_z = 1) in;
#endif

#if USE_BDA == 0
	layout(binding = 0, std430) buffer src_buf { realC src[]; };
	layout(binding = 1, std430) buffer dest_buf { realC dest[]; };
#endif

layout(push_constant) uniform CopyMatrixFast
{
	int ld;
#if USE_BDA
	__global const realC* restrict src,
	__global realC* dest,
#endif
	real_arg arg_alpha;
} args;

void main()
{
	if (args.ld % COPY_VW != 0) return;

	const real alpha = GetRealArg(args.arg_alpha);
	UNROLL(COPY_WPT)
	for (int _w_one = 0; _w_one < COPY_WPT; _w_one += 1) {
		const int id_one = get_global_id(0);
		const int id_two = (get_group_id(1)*COPY_WPT + _w_one) * COPY_DIMY + get_local_id(1);
		const int id = id_two*(args.ld/COPY_VW) + id_one;
		realC result;
		#if COPY_VW == 1
			Multiply(result, alpha, src[id]);
		#else
			vsMultiply(result, alpha, src[id], COPY_VW);
		#endif
		dest[id] = result;
	}
}

// =================================================================================================

// End of the C++11 raw string literal
//)"

// =================================================================================================
