#version 450
#include "transpose_matrix_impl.glsl"
#ifndef ROUTINE_GEMMSTRIDEDBATCHED
	#define ROUTINE_GEMMSTRIDEDBATCHED
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

// =================================================================================================
#if 1

// Strided-batched version of the above
layout(push_constant) uniform TransposeMatrixStridedBatched
{
	int src_one; int src_two;
	int src_ld; int src_offset;
	int src_stride;
#if USE_BDA
	__global real* restrict src;
#endif
	int dest_one; int dest_two;
	int dest_ld; int dest_offset;
	int dest_stride;
#if USE_BDA
	__global real* dest
#endif
} args;

void main()
{
	const int batch = get_group_id(2);
	const int src_offset_batch = args.src_offset + args.src_stride * batch;
	const int dest_offset_batch = args.dest_offset + args.dest_stride * batch;
	real alpha; SetToOne(alpha);

	_TransposeMatrix(
		//tile,
		args.src_one, args.src_two, args.src_ld, src_offset_batch,
#if USE_BDA
		src,
#endif
		args.dest_one, args.dest_two, args.dest_ld, dest_offset_batch,
#if USE_BDA
		dest,
#endif
		alpha, 0, 0, 0);
}

#endif
// =================================================================================================

// End of the C++11 raw string literal
//)"

// =================================================================================================
