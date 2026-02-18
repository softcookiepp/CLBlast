#version 450





// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the common kernels shared among different BLAS functions. This file contains
// kernels to copy and pad matrices in various ways, including:
// 1) copying into a larger matrix by adding padding
// 2) copying into a smaller matrix by optionally removing padding. This is the general version
//		without restrictions, see the 'copy.opencl' file for a faster but more restricted copy kernel.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
//R"(
#ifndef ROUTINE_GEMMBATCHED
	#define ROUTINE_GEMMBATCHED
#endif
#include "copy_matrix_batched_impl.glsl"
// =================================================================================================
#if 1//def ROUTINE_GEMMBATCHED

// Batched version of the above
#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = PAD_DIMX, local_size_y = PAD_DIMY, local_size_z = 1) in;
#endif

layout(push_constant) uniform CopyMatrixBatched
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
};

void main()
{
	const int batch = get_group_id(2);
	const int src_offset = src_offsets[batch];
	const int dest_offset = dest_offsets[batch];
	real alpha; SetToOne(alpha);
	_CopyMatrix(src_one, src_two, src_ld, src_offset,
#if USE_BDA
		src,
#endif
		dest_one, dest_two, dest_ld, dest_offset,
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
