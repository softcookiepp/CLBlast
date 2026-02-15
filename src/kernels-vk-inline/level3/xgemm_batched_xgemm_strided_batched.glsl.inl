
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the batched version of the non-direct GEMM kernel. See part 1 for information
// about the non-batched version of the kernel.
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
R"(
// =================================================================================================

#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = MDIMC, local_size_y = NDIMC, local_size_z = 1) in;
#endif

layout(push_constant, std430) uniform XgemmStridedBatched
{
	int kSizeM; int kSizeN; int kSizeK;
	real_arg arg_alpha; real_arg arg_beta;
#if USE_BDA
	__global realM* restrict agm;
#endif
	int a_one; int a_two;
#if USE_BDA
	__global realN* restrict bgm;
#endif
	int b_one; int b_two;
#if USE_BDA
	__global realM* cgm;
#endif
	int c_one; int c_two;
} args;

void main()
{
	const int batch = get_group_id(2);
	const real alpha = GetRealArg(args.arg_alpha);
	const real beta = GetRealArg(args.arg_beta);

	// Sets the offsets
	const int a_offset = batch * args.a_one * args.a_two;
	const int b_offset = batch * args.b_one * args.b_two;
	const int c_offset = batch * args.c_one * args.c_two;
#if USE_BDA
	const __global realM* restrict agm_ = &agm[a_offset / VWM];
	const __global realN* restrict bgm_ = &bgm[b_offset / VWN];
	__global realM* restrict cgm_ = &cgm[c_offset / VWM];
#endif

	// Computes the matrix-multiplication and stores the result in global memory
	XgemmBody(args.kSizeM, args.kSizeN, args.kSizeK,
#if USE_BDA
		agm_, bgm_, cgm_,
#else
		a_offset, b_offset, c_offset,
#endif
		alpha, beta
	);
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
