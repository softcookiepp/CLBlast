#version 450
#ifndef ROUTINE_GEMMBATCHED
	#define ROUTINE_GEMMBATCHED
#endif

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
#include "transpose_pad_matrix_batched_impl.glsl"
// =================================================================================================
#if 1//def ROUTINE_GEMMBATCHED

// Batched version of the above
layout(push_constant) uniform TransposePadMatrixBatched
{
	int src_one; int src_two;
	int src_ld;
#if USE_BDA
	__constant int* src_offsets;
	__global real* restrict src;
#endif
	int dest_one; int dest_two;
	int dest_ld;
#if USE_BDA
	__constant int* dest_offsets;
	__global real* dest;
#endif
	int do_conjugate;
} args;

void main()
{
	const int batch = get_group_id(2);
	const int src_offset = src_offsets[batch];
	const int dest_offset = dest_offsets[batch];
	real alpha; SetToOne(alpha);

	_TransposePadMatrix(//tile,
		args.src_one, args.src_two, args.src_ld, src_offset,
#if USE_BDA
		src,
#endif
		args.dest_one, args.dest_two, args.dest_ld, dest_offset,
#if USE_BDA
		dest,
#endif
		alpha, args.do_conjugate);
}

#endif
// =================================================================================================

// End of the C++11 raw string literal
//)"

// =================================================================================================
