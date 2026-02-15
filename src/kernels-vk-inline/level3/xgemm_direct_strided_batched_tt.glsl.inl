
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the batched version of the direct GEMM kernels. See part 1 for information
// about the non-batched version of the kernel.
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
R"(
// =================================================================================================

// =================================================================================================

// Direct version of the strided-batched GEMM kernel with [A, B] = [transposed, transposed]
#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = MDIMCD, local_size_y = NDIMCD, local_size_z = 1) in;
#endif
layout(push_constant) uniform XgemmDirectStridedBatchedTT
{
	int kSizeM; int kSizeN; int kSizeK;
	real_arg arg_alpha; real_arg arg_beta;
#if USE_BDA
	__global realMD* restrict agm;
#endif
	int a_offset; int a_ld; int a_stride;
#if USE_BDA
	__global realND* restrict bgm;
#endif
	int b_offset; int b_ld; int b_stride;
#if USE_BDA
	__global real* cgm;
#endif
	int c_offset; int c_ld; int c_stride;
	int c_transpose; int a_conjugate; int b_conjugate;
} args;

void main()																 
{
	const int batch = get_group_id(2);
	const int a_offset_batch = args.a_offset + args.a_stride * batch;
	const int b_offset_batch = args.b_offset + args.b_stride * batch;
	const int c_offset_batch = args.c_offset + args.c_stride * batch;

	XgemmDirect(args.kSizeM, args.kSizeN, args.kSizeK, args.arg_alpha, args.arg_beta,
#if USE_BDA
		agm,
#endif
		a_offset_batch, args.a_ld,
#if USE_BDA
		bgm,
#endif
		b_offset_batch, args.b_ld,
#if USE_BDA
		cgm,
#endif
		c_offset_batch, args.c_ld,
		//alm, blm,
		1, 1, args.c_transpose, args.a_conjugate, args.b_conjugate);
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
