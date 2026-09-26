#version 450

// =================================================================================================
// This file is part of the CLBlast project. This file contains the col2im kernel, taken from:
// https://gist.github.com/vbkaisetsu/a98299df827f9a5245635f646c1d94be
// Credits go to https://github.com/vbkaisetsu
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
//R"(
#include "../common.glsl"
#define ROUTINE_IM2COL 0
#include "im2col_col2im_common.glsl"

layout(push_constant) uniform Xcol2imKernelNormal
{
	int input_h; int input_w; int channels;
	int output_h; int output_w;
	int kernel_h; int kernel_w;
	int pad_h; int pad_w;
	int stride_h; int stride_w;
	int dilation_h; int dilation_w;
	int stride_bez_h; int stride_bez_w;
	int dilation_bez_h; int dilation_bez_w;
	int gcd_h; int gcd_w;
#if USE_BDA
	__global real* restrict col_buffer;
#endif
	int col_offset;
#if USE_BDA
	__global real* im_buffer;
#endif
	int im_offset;
};

void main()
{
	const bool kernel_flip = KERNEL_FLIP;
	Xcol2im(input_h, input_w, channels, output_h, output_w, kernel_h, kernel_w,
					pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
					stride_bez_h, stride_bez_w, dilation_bez_h, dilation_bez_w, gcd_h, gcd_w,
					kernel_flip,
#if USE_BDA
					col_buffer,
#endif
					col_offset,
#if USE_BDA
					im_buffer,
#endif
					im_offset);
}

// =================================================================================================

// End of the C++11 raw string literal
//)"

// =================================================================================================
