#version 450

// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 3 of 3 of the GEMM kernel. See part 1 for more information.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
//R"(
#include "xgemm_direct_part3.glsl"
// =================================================================================================

// Direct version of the GEMM kernel with [A, B] = [transposed, transposed]
// somehow the args are shared across the entire thingy
layout(push_constant) uniform XgemmDirectTT
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
};


void main()
{
	XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
#if USE_BDA
		agm,
#endif
		a_offset, a_ld,
#if USE_BDA
		bgm,
#endif
		b_offset, b_ld,
#if USE_BDA
		cgm,
#endif
		c_offset, c_ld,
		//alm, blm,
		1, 1, c_transpose, a_conjugate, b_conjugate);
}

// =================================================================================================

// End of the C++11 raw string literal
//)"

// =================================================================================================
