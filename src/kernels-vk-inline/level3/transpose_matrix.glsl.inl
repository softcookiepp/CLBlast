
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

// literal). Comment-out this line for syntax-highlighting when developing.
R"(


// =================================================================================================

// Interface to the above function
layout(push_constant) uniform TransposeMatrix
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
	_TransposeMatrix(
		//tile,
		args.src_one, args.src_two, args.src_ld, args.src_offset,
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
