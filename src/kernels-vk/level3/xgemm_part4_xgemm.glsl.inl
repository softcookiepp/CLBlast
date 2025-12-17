// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 4 of 4 of the GEMM kernel. See part 1 for more information.
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// Main entry point of the kernel. This is the regular full version.
#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = MDIMC, local_size_y = NDIMC, local_size_z = 1) in;
#endif

// buffers already defined in part 1

layout(push_constant, std430) uniform Xgemm
{
	int kSizeM; int kSizeN; int kSizeK;
	real_arg arg_alpha;
	real_arg arg_beta;
#if USE_BDA
	__global realM* restrict agm;
	__global realN* restrict bgm;
	__global realM* cgm;
#endif
	int b_offset; int c_offset;
} args;

void main()
{
	const real alpha = GetRealArg(args.arg_alpha);
	const real beta = GetRealArg(args.arg_beta);

	// Adds the offsets (in case of use of a single temporary buffer for A, B, and C)
#if USE_BDA
	// not allowed without BDA; plus BDA use isn't yet implemented
	bgm = &bgm[b_offset];
	cgm = &cgm[c_offset];
#endif

	// Computes the matrix-multiplication and stores the result in global memory
	XgemmBody(args.kSizeM, args.kSizeN, args.kSizeK,
#if USE_BDA
		agm, bgm, cgm,
#else
		0, args.b_offset, args.c_offset,
#endif
		alpha, beta
	);
}

)"
// End of the C++11 raw string literal

// =================================================================================================
