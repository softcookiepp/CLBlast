#version 450

// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the common kernels shared among different BLAS functions. This file contains
// kernels to transpose matrices in various ways, including:
// 1) transposing into a larger matrix by adding padding
// 2) transposing into a smaller matrix by optionally removing padding. This is the general version
//		without restrictions, see the 'transpose.opencl' file for a faster but more restricted
//		transpose kernel.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
//R"(
#include "transpose_pad_matrix_impl.glsl"
// =================================================================================================

// Interface to the above function
layout(push_constant) uniform TransposePadMatrix
{
	int src_one; int src_two;
	int src_ld; int src_offset;
#if USE_BDA
	__global real* restrict src;
#endif
	int dest_one; int dest_two;
	int dest_ld; int dest_offset;
#if USE_BDA
	__global real* dest;
#endif
	real_arg arg_alpha;
	int do_conjugate;
};

void main()
{
	const real alpha = GetRealArg(arg_alpha);
	
	_TransposePadMatrix(//tile,
		src_one, src_two, src_ld, src_offset,
#if USE_BDA
		src,
#endif
		dest_one, dest_two, dest_ld, dest_offset,
#if USE_BDA
		dest,
#endif
		alpha, do_conjugate);
}


// =================================================================================================

// End of the C++11 raw string literal
//)"

// =================================================================================================
