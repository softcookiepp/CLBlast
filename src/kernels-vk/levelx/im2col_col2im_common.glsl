#ifndef IM2COL_COL2IM_COMMON
#define IM2COL_COL2IM_COMMON

// Work-group size parameters re-used from the 'copy' kernel

#ifdef COPY_DIMX
	#undef COPY_DIMX
#endif
layout(constant_id = 0) const int COPY_DIMX = 8; // Local workgroup size in the first dimension (w)

#ifdef COPY_DIMY
	#undef COPY_DIMY
#endif
layout(constant_id = 1) const int COPY_DIMY = 8; // Local workgroup size in the second dimension (h)

layout(constant_id = 2) const bool KERNEL_FLIP = false; // whether or not to use the flip kernel

// =================================================================================================

// buffer defs
#if USE_BDA == 0
	#ifndef ROUTINE_IM2COL
		#define ROUTINE_IM2COL 0
	#endif
	#if ROUTINE_IM2COL == 1
		layout(binding = 0, std430) readonly buffer im_buffer_def { real im_buffer[]; }; 
		layout(binding = 1, std430) writeonly buffer col_buffer_def { real col_buffer[]; };
	#else
		layout(binding = 0, std430) buffer col_buffer_buf { real col_buffer[]; };
		layout(binding = 1, std430) buffer im_buffer_buf { real im_buffer[]; };
	#endif
#endif

#if ROUTINE_IM2COL == 1
// Main body of the kernel
void Xim2col(const int input_h, const int input_w, const int channels,
	const int output_h, const int output_w,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	const bool kernel_flip,
	#if USE_BDA
		const __global real* restrict im_buffer,
	#endif
		const int im_offset,
	#if USE_BDA
		__global real* col_buffer,
	#endif
	const int col_offset)
{

	// Thread IDs
	const int w_id = get_global_id(0); // image width, max 'output_w'
	const int h_id = (get_global_id(1)) % output_h; // image height, max 'output_h'
	const int c_id = (get_global_id(1)) / output_h; // input channels
	if (h_id < output_h && w_id < output_w && c_id < channels) {

		for (int kh_id = 0; kh_id < kernel_h; ++kh_id) { // kernel height
			for (int kw_id = 0; kw_id < kernel_w; ++kw_id) { // kernel width

				// Retrieves the input value
				const int h_index = -pad_h + kh_id * dilation_h + stride_h * h_id;
				const int w_index = -pad_w + kw_id * dilation_w + stride_w * w_id;
				real val;
				if (h_index >= 0 && h_index < input_h &&
						w_index >= 0 && w_index < input_w) {
					const int input_index = w_index + input_w * (h_index + input_h * c_id);
					val = im_buffer[input_index + im_offset];
				}
				else {
					SetToZero(val);
				}

				// Sets the output value
				const int kernel_index = (kernel_flip)
															 ? kernel_h * kernel_w - kw_id - kernel_w * kh_id - 1
															 : kw_id + kernel_w * kh_id;
				const int patch_index = w_id + output_w * h_id;
				const int output_index = patch_index + kernel_index * output_w * output_h +
																	c_id * output_w * output_h * kernel_h * kernel_w;
				col_buffer[output_index + col_offset] = val;
			}
		}
	}
}
#else
int grid_ceil(const int x, const int step)
{
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
#endif

// =================================================================================================

// Kernel flip version of the Xim2col kernel (for convolution)
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z = 1) in;
#endif
