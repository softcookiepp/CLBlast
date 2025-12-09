#version 450
#include "../common.glsl"
// =================================================================================================
// This file is part of the CLBlast project. This file contains the col2im kernel, taken from:
// https://gist.github.com/vbkaisetsu/a98299df827f9a5245635f646c1d94be
// Credits go to https://github.com/vbkaisetsu
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
//R"(

// Work-group size parameters re-used from the 'copy' kernel
#ifndef COPY_DIMX
	#define COPY_DIMX 8			// Local workgroup size in the first dimension (w)
#endif
#ifndef COPY_DIMY
	#define COPY_DIMY 8			// Local workgroup size in the second dimension (h)
#endif

// =================================================================================================

// buffer declarations
#if USE_BDA == 0
	layout(binding = 0, std430) buffer col_buffer_buf { real col_buffer[]; };
	layout(binding = 1, std430) buffer im_buffer_buf { real im_buffer[]; };
#endif

int grid_ceil(const int x, const int step) {
	return x > 0 ? ((x - 1) / step + 1) * step : x / step * step;
}

// Main body of the kernel
void Xcol2im(const int input_h, const int input_w, const int channels,
	const int output_h, const int output_w,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	const int stride_bez_h, const int stride_bez_w,
	const int dilation_bez_h, const int dilation_bez_w,
	const int gcd_h, const int gcd_w,
	const bool kernel_flip,
#if USE_BDA
	const __global real* restrict col_buffer,
#endif
	const int col_offset,
#if USE_BDA
	__global real* im_buffer,
#endif
	const int im_offset)
{

	const int input_h_scaled = (input_h - 1) / gcd_h + 1;

	// Thread IDs
	const int gcd_scale_w = get_global_id(0) + (pad_w - 1) / gcd_w + 1;
	const int gcd_scale_h = get_global_id(1) % input_h_scaled + (pad_h - 1) / gcd_h + 1;
	const int c_id = get_global_id(1) / input_h_scaled;

	const int w_index = gcd_scale_w * gcd_w - pad_w;
	const int h_index = gcd_scale_h * gcd_h - pad_h;
	const int th_step = stride_h * dilation_h / gcd_h;
	const int th_begin = grid_ceil(max(-stride_bez_h * gcd_scale_h * stride_h,
																		 (dilation_bez_h * gcd_scale_h - kernel_h + 1) * dilation_h),
																 th_step);
	const int th_end = min((output_h - stride_bez_h * gcd_scale_h) * stride_h,
												 (dilation_bez_h * gcd_scale_h + 1) * dilation_h);
	const int tw_step = stride_w * dilation_w / gcd_w;
	const int tw_begin = grid_ceil(max(-stride_bez_w * gcd_scale_w * stride_w,
																		 (dilation_bez_w * gcd_scale_w - kernel_w + 1) * dilation_w),
																 tw_step);
	const int tw_end = min((output_w - stride_bez_w * gcd_scale_w) * stride_w,
												 (dilation_bez_w * gcd_scale_w + 1) * dilation_w);
	if (w_index < input_w && c_id < channels) {
		real val;
		SetToZero(val);
		for (int th = th_begin; th < th_end; th += th_step) {
			for (int tw = tw_begin; tw < tw_end; tw += tw_step) {
				const int kh_id = -th / dilation_h + dilation_bez_h * gcd_scale_h;
				const int kw_id = -tw / dilation_w + dilation_bez_w * gcd_scale_w;
				const int h_id = th / stride_h + stride_bez_h * gcd_scale_h;
				const int w_id = tw / stride_w + stride_bez_w * gcd_scale_w;
				const int kernel_index = (kernel_flip)
															 ? kernel_h * kernel_w - kw_id - kernel_w * kh_id - 1
															 : kw_id + kernel_w * kh_id;
				const int patch_index = w_id + output_w * h_id;
				const int output_index = patch_index + kernel_index * output_w * output_h +
																 c_id * output_w * output_h * kernel_h * kernel_w;
				Add(val, val, col_buffer[output_index + col_offset]);
			}
		}

		// Accumulates the resulting value with the existing im-buffer (+= val)
		const int input_index = w_index + input_w * (h_index + input_h * c_id);
		real im_buffer_value = im_buffer[input_index + im_offset];
		Add(im_buffer[input_index + im_offset], im_buffer_value, val);
	}
}


// =================================================================================================

// Kernel flip version of the Xcol2im kernel (for convolution)
#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = COPY_DIMX, local_size_y = COPY_DIMY, local_size_z = 1) in;
#endif

layout(push_constant) uniform Xcol2imKernelFlip
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
} args;

void main()
{
	const bool kernel_flip = true;
	Xcol2im(args.input_h, args.input_w, args.channels, args.output_h, args.output_w, args.kernel_h, args.kernel_w,
					args.pad_h, args.pad_w, args.stride_h, args.stride_w, args.dilation_h, args.dilation_w,
					args.stride_bez_h, args.stride_bez_w, args.dilation_bez_h, args.dilation_bez_w, args.gcd_h, args.gcd_w,
					kernel_flip,
#if USE_BDA
					col_buffer,
#endif
					args.col_offset,
#if USE_BDA
					im_buffer,
#endif		
					args.im_offset);
}

// =================================================================================================

// End of the C++11 raw string literal
//)"

// =================================================================================================
