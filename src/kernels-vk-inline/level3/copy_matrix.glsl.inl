
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

// Interface to the above function
#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = PAD_DIMX, local_size_y = PAD_DIMY, local_size_z = 1) in;
#endif

layout(push_constant) uniform CopyMatrix
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
	int upper; int lower;
	int diagonal_imag_zero;
} args;

void main()
{
	const real alpha = GetRealArg(args.arg_alpha);
	_CopyMatrix(args.src_one, args.src_two, args.src_ld, args.src_offset,
#if USE_BDA
		src,
#endif
		args.dest_one, args.dest_two, args.dest_ld, args.dest_offset,
#if USE_BDA
		dest,
#endif
		alpha, args.upper, args.lower, args.diagonal_imag_zero);
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
