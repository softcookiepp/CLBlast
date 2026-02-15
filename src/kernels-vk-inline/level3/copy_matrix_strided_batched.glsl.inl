
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

// literal). Comment-out this line for syntax-highlighting when developing.
R"(
// =================================================================================================

#if 1 // defined(ROUTINE_GEMMSTRIDEDBATCHED)

// Strided-batched version of the above
// Batched version of the above
#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = PAD_DIMX, local_size_y = PAD_DIMY, local_size_z = 1) in;
#endif

layout(push_constant) uniform CopyMatrixStridedBatched
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
	_CopyMatrix(args.src_one, args.src_two, args.src_ld, src_offset_batch,
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
)"

// =================================================================================================
