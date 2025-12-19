
// DO NOT USE THIS FILE YA SILLY
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 3 of 3 of the GEMM kernel. See part 1 for more information.
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================

// global and shared memory declarations go here, since they are shared across all kernels c:
// *gm and *gms both bind to the same underlying memory
layout(binding = 0) buffer agm_buf { realMD agm[]; };
layout(binding = 1) buffer bgm_buf { realND bgm[]; };
layout(binding = 2) buffer cgm_buf { real cgm[]; };

layout(binding = 3) buffer agms_buf { real agms[]; };
layout(binding = 4) buffer bgms_buf { real bgms[]; };

shared real alm[WGD * (WGD + PADA)];
shared real blm[WGD * (WGD + PADB)];

#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = MDIMCD, local_size_y = NDIMCD, local_size_z = 1) in;
#endif

// somehow the args are shared across the entire thingy
layout(push_constant) uniform XgemmDirectArgs
{
	int kSizeM; int kSizeN; int kSizeK;
	real_arg arg_alpha; real_arg arg_beta;
#if USE_BDA
	__global realMD* restrict agm;
#endif
	int a_offset; int a_ld;
#if USE_BDA
	__global realND* restrict bgm;
#endif
	int b_offset; int b_ld;
#if USE_BDA
	__global real* cgm;
#endif
	int c_offset; int c_ld;
	int c_transpose; int a_conjugate; int b_conjugate;
} args;

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
