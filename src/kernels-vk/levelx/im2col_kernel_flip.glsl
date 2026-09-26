#version 450

// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the im2col kernel.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
//R"(
#include "../common.glsl"
#define ROUTINE_IM2COL 1
#include "im2col_col2im_common.glsl"

layout(push_constant) uniform Xim2colKernelFlip
{
	int input_h; 	int input_w; 	int channels;
	int output_h; 	int output_w;
	int kernel_h; 	int kernel_w;
	int pad_h; 	int pad_w;
	int stride_h; 	int stride_w;
	int dilation_h; 	int dilation_w;
#if USE_BDA
	__global real* restrict im_buffer; 
#endif
	int im_offset;
#if USE_BDA
	__global real* col_buffer;
#endif
	int col_offset;
};

void main()
{
	const bool kernel_flip = KERNEL_FLIP;
	Xim2col(input_h, input_w, channels, output_h, output_w, kernel_h, kernel_w,
					pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
					kernel_flip,
#if USE_BDA
					im_buffer,
#endif
					im_offset,
#if USE_BDA
					col_buffer,
#endif
					col_offset);
}

// =================================================================================================

// End of the C++11 raw string literal
//)"

// =================================================================================================
