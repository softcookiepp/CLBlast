

// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the an implementation of 3D convolution on a 4D image using GEMM kernels. It
// uses parameters from the direct GEMM kernel. This is the part with the loads from memory (1/2).
// This uses "CONVGEMM_WITH_IM2COL" as a switch to select between direct convgemm or first running
// the im2col kernel to create a 'col' temporary matrix.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
//R"(
#include "../level3/xgemm_direct_part2.glsl"
// =================================================================================================

// Loads global off-chip memory into thread-private register files. This function is specific for
// loading the image input tensor. This includes a bounds check.
// result should be of `real` type
#define GlobalToPrivateCheckedImage(result, imagegm, image_offset_batch, h_id, w_id, kwg, input_h, input_w, channels, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, kernel_flip) \
{ \
	const int kernel_2d_index = kwg % (kernel_h * kernel_w); \
	const int kw_id = (kernel_flip) \
									? kernel_w - kernel_2d_index % kernel_w - 1 \
									: kernel_2d_index % kernel_w; \
	const int kh_id = (kernel_flip) \
									? kernel_h - kernel_2d_index / kernel_w - 1 \
									: kernel_2d_index / kernel_w; \
	const int c_id = kwg / (kernel_h * kernel_w); \
	const int h_index = -pad_h + kh_id * dilation_h + stride_h * h_id; \
	const int w_index = -pad_w + kw_id * dilation_w + stride_w * w_id; \
	if (h_index >= 0 && h_index < input_h && \
			w_index >= 0 && w_index < input_w) { \
		const int image_index = w_index + input_w * (h_index + input_h * c_id); \
		result = imagegm[image_index + image_offset_batch]; \
	} \
	else { \
		SetToZero(result); \
	} \
}

// get indices for the below function
ivec2 get_la0_la1()
{
	#if MDIMCD == MDIMAD
		return ivec2(get_local_id(0), get_local_id(1));
		// const int la0 = get_local_id(0);
		// const int la1 = get_local_id(1);
	#else
		const int tid = get_local_id(0) + MDIMCD*get_local_id(1);
		//const int la0 = tid % MDIMAD;
		//const int la1 = tid / MDIMAD;
		return ivec2(tid % MDIMAD, tid / MDIMAD);
	#endif
}

// Loads global off-chip memory into local (shared) memory on-chip. This function is specific for
// loading the image input tensor. This includes a bounds check.
// this is atrocious, but its literally the only way to do this in GLSL
#define GlobalToLocalCheckedImage(imagegm, alm, image_offset_batch, output_w, kwg, input_h, input_w, channels, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, kernel_flip) \
{ \
	const ivec2 la0la1 = get_la0_la1(); \
	const int la0 = la0la1.x; \
	const int la1 = la0la1.y; \
	for (int _mia = 0; _mia < MWAD; _mia += 1) { \
		for (int _kia = 0; _kia < KWAD; _kia += 1) { \
			int mg = _mia + la0*MWAD; \
			int kg = _kia + la1*KWAD; \
			int idm = mg + GetGroupID0()*WGD; \
			int idk = kg + kwg; \
			const int w_id = idm % output_w; \
			const int h_id = idm / output_w; \
			const int kernel_2d_index = idk % (kernel_h * kernel_w); \
			const int kw_id = (kernel_flip) \
											? kernel_w - kernel_2d_index % kernel_w - 1 \
											: kernel_2d_index % kernel_w; \
			const int kh_id = (kernel_flip) \
											? kernel_h - kernel_2d_index / kernel_w - 1 \
											: kernel_2d_index / kernel_w; \
			const int c_id = idk / (kernel_h * kernel_w); \
			const int h_index = -pad_h + kh_id * dilation_h + stride_h * h_id; \
			const int w_index = -pad_w + kw_id * dilation_w + stride_w * w_id; \
			if (h_index >= 0 && h_index < input_h && \
					w_index >= 0 && w_index < input_w) { \
				const int image_index = w_index + input_w * (h_index + input_h * c_id); \
				const real result = imagegm[image_index + image_offset_batch]; \
				alm[kg*(WGD + PADA) + mg] = result; \
			} \
			else { \
				SetToZero(alm[kg*(WGD + PADA) + mg]); \
			} \
		} \
	} \
}

// =================================================================================================

// End of the C++11 raw string literal
//)"

// =================================================================================================
