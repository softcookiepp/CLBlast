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

// declare buffers ahead of time, since this function uses them and they share the same names amongst the kernels that use it
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


// Same as above, but now un-pads a matrix. This kernel reads data from a padded source matrix, but
// writes only the actual data back to the destination matrix. Again, the ld value and offset can
// be different.
void _CopyMatrix(const int src_one, const int src_two,
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
		const int upper, const int lower,
		const int diagonal_imag_zero)
{
	// Loops over the work per thread in both dimensions
	// #pragma unroll
	for (int _w_one = 0; _w_one < PAD_WPTX; _w_one += 1) {
		const int id_one = (get_group_id(0)*PAD_WPTX + _w_one) * PAD_DIMX + get_local_id(0);
		// #pragma unroll
		for (int _w_two = 0; _w_two < PAD_WPTY; _w_two += 1) {
			const int id_two = (get_group_id(1)*PAD_WPTY + _w_two) * PAD_DIMY + get_local_id(1);

			// Masking in case of triangular matrices: updates only the upper or lower part
			bool condition = true;
			#if defined(ROUTINE_SYRK) || defined(ROUTINE_HERK) || defined(ROUTINE_SYR2K) || defined(ROUTINE_HER2K)
				if (upper == 1) { condition = (id_two >= id_one); }
				else if (lower == 1) { condition = (id_two <= id_one); }
			#endif
			if (condition) {

				// Copies the value into the destination matrix. This is always within bounds of the source
				// matrix, as we know that the destination matrix is smaller or equal to the source.
				if (id_two < dest_two && id_one < dest_one) {
					real value = src[id_two*src_ld + id_one + src_offset];
					if (diagonal_imag_zero == 1 && id_one == id_two) { ImagToZero(value); }
					Multiply(dest[id_two*dest_ld + id_one + dest_offset], alpha, value);
				}
			}
		}
	}
}

// End of the C++11 raw string literal
)"

// =================================================================================================
