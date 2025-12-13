// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the common kernels shared among different BLAS functions. This file contains
// kernels to copy and pad matrices in various ways, including:
// 1) copying into a larger matrix by adding padding
// 2) copying into a smaller matrix by optionally removing padding. This is the general version
//		without restrictions, see the 'copy.opencl' file for a faster but more restricted copy kernel.
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================

#if USE_BDA == 0
	#if 0
		layout(binding = 0, std430) buffer src_offsets_buf { int src_offsets[]; };
		layout(binding = 1, std430) buffer src_buf { real src[]; };
		layout(binding = 2, std430) buffer dest_offsets_buf { int dest_offsets[]; };
		layout(binding = 3, std430) buffer dest_buf { real dest[]; };
	#else
		layout(binding = 0, std430) buffer src_buf { real src[]; };
		layout(binding = 1, std430) buffer dest_buf { real dest[]; };
	#endif
#endif

// Copies a matrix from source to destination. The output is padded with zero values in case the
// destination matrix dimensions are larger than the source matrix dimensions. Additionally, the ld
// value and offset can be different.
void _CopyPadMatrix(const int src_one, const int src_two,
	const int src_ld, const int src_offset,
#if USE_BDA
	__global const real* restrict src,
#endif
	const int dest_one, const int dest_two,
	const int dest_ld, const int dest_offset,
#if USE_BDA
	__global real* dest,
#endif
	const real alpha,
	const int do_conjugate)
{
	// Loops over the work per thread in both dimensions
	// #pragma unroll
	for (int _w_one = 0; _w_one < PAD_WPTX; _w_one += 1) {
		const int id_one = (get_group_id(0)*PAD_WPTX + _w_one) * PAD_DIMX + get_local_id(0);
		// #pragma unroll
		for (int _w_two = 0; _w_two < PAD_WPTY; _w_two += 1) {
			const int id_two = (get_group_id(1)*PAD_WPTY + _w_two) * PAD_DIMY + get_local_id(1);
			if (id_two < dest_two && id_one < dest_one) {

				// Loads data if the thread IDs are within bounds of the source matrix. Otherwise, set the
				// value to be written to zero.
				real value;
				SetToZero(value);
				if (id_two < src_two && id_one < src_one) {
					value = src[id_two*src_ld + id_one + src_offset];
				}

				// Stores the value in the destination matrix
				if (do_conjugate == 1) { COMPLEX_CONJUGATE(value); }
				Multiply(dest[id_two*dest_ld + id_one + dest_offset], alpha, value);
			}
		}
	}
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
