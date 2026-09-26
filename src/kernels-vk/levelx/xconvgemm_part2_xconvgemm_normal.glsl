#version 450

// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the an implementation of 3D convolution on a 4D image using GEMM kernels. It
// uses parameters from the direct GEMM kernel. This part contains the main kernel (2/2).
// This uses "CONVGEMM_WITH_IM2COL" as a switch to select between direct convgemm or first running
// the im2col kernel to create a 'col' temporary matrix.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
//R"(
#include "xconvgemm_part2_xconvgemm_function.glsl"
// =================================================================================================

layout(push_constant, std430) uniform XconvgemmNormal
{
	int num_patches; int num_kernels; int patch_size;
#if USE_BDA
	__global realND* restrict kernelgm;
#endif
	int kernel_offset;
#if USE_BDA
	__global real* resultgm;
#endif
	int result_offset; int result_stride;
#if USE_BDA
	__global realMD* restrict imagegm;
#endif
	int image_offset;
	int input_h; int input_w; int channels;
	int kernel_h; int kernel_w;
	int pad_h; int pad_w;
	int stride_h; int stride_w;
	int dilation_h; int dilation_w;
	int output_h; int output_w;
};

void main()										 
{
	const bool kernel_flip = false;
	Xconvgemm(num_patches, num_kernels, patch_size,
#if USE_BDA
		kernelgm,
#endif
		kernel_offset,
#if USE_BDA
		resultgm,
#endif
		result_offset, result_stride,
#if USE_BDA
		imagegm,
#endif
		image_offset,
		input_h, input_w, channels,
		kernel_h, kernel_w,
		pad_h, pad_w,
		stride_h, stride_w,
		dilation_h, dilation_w,
		output_h, output_w,
		//alm, blm,
		kernel_flip);
}


// =================================================================================================

// End of the C++11 raw string literal
//)"

// =================================================================================================
