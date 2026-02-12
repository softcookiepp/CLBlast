
R"(
// Direct version of the batched GEMM kernel with [A, B] = [non-transposed, non-transposed]
#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = MDIMCD, local_size_y = NDIMCD, local_size_z = 1) in;
#endif
layout(push_constant, std430) uniform XgemmDirectBatchedNN
{
	int kSizeM; int kSizeN; int kSizeK;
#if USE_BDA
	__constant real_arg* arg_alphas; __constant real_arg* arg_betas;
	__global realMD* restrict agm; __constant int* a_offsets;
#endif
	int a_ld;
#if USE_BDA
	__global realND* restrict bgm; __constant int* b_offsets;
#endif
	int b_ld;
#if USE_BDA
	__global real* cgm; __constant int* c_offsets;
#endif
	int c_ld;
	int c_transpose; int a_conjugate; int b_conjugate;
} args;

void main()
{
	const int batch = get_group_id(2);
	const real_arg arg_alpha = arg_alphas[batch];
	const real_arg arg_beta = arg_betas[batch];
	const int a_offset = a_offsets[batch];
	const int b_offset = b_offsets[batch];
	const int c_offset = c_offsets[batch];

	XgemmDirect(args.kSizeM, args.kSizeN, args.kSizeK, arg_alpha, arg_beta,
#if USE_BDA
		agm,
#endif
		a_offset, args.a_ld,
#if USE_BDA
		bgm,
#endif
		b_offset, args.b_ld,
#if USE_BDA
		cgm,
#endif
		c_offset, args.c_ld,
		//alm, blm,
		0, 0, args.c_transpose, args.a_conjugate, args.b_conjugate);
}
)"
