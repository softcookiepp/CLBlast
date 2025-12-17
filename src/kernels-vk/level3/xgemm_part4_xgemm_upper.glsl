#version 450
#include "xgemm_part3.glsl"
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 4 of 4 of the GEMM kernel. See part 1 for more information.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
//R"(

// The upper-triangular and lower-triangular kernels are only used in special cases

// Main entry point of the kernel. This is the upper-triangular version.
#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = MDIMC, local_size_y = NDIMC, local_size_z = 1) in;
#endif

layout(push_constant, std430) uniform XgemmUpper
{
	int kSizeN; int kSizeK;
	real_arg arg_alpha;
	real_arg arg_beta;
#if USE_BDA
	__global realM* restrict agm;
	__global realN* restrict bgm;
	__global realM* cgm;
#endif
} args;

void main()
{
	const real alpha = GetRealArg(args.arg_alpha);
	const real beta = GetRealArg(args.arg_beta);

	// Skip these threads if they do not contain threads contributing to the upper-triangle
	if ((GetGroupID1() + 1)*NWG < GetGroupID0()*MWG) {
		return;
	}

	// Computes the matrix-multiplication and stores the result in global memory
	XgemmBody(args.kSizeN, args.kSizeN, args.kSizeK,
#if USE_BDA
		agm, bgm, cgm,
#else
		0, 0, 0,
#endif
		alpha, beta);
}


//)"
// End of the C++11 raw string literal

// =================================================================================================
