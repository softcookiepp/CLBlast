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
} args;

void main()										 
{
	const bool kernel_flip = false;
	Xconvgemm(args.num_patches, args.num_kernels, args.patch_size,
#if USE_BDA
		kernelgm,
#endif
		args.kernel_offset,
#if USE_BDA
		resultgm,
#endif
		args.result_offset, args.result_stride,
#if USE_BDA
		imagegm,
#endif
		args.image_offset,
		args.input_h, args.input_w, args.channels,
		args.kernel_h, args.kernel_w,
		args.pad_h, args.pad_w,
		args.stride_h, args.stride_w,
		args.dilation_h, args.dilation_w,
		args.output_h, args.output_w,
		//alm, blm,
		kernel_flip);
}


// =================================================================================================

// End of the C++11 raw string literal
//)"

// =================================================================================================
