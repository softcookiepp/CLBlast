
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the common kernels shared among different BLAS functions. This file contains
// kernels to transpose matrices in various ways, including:
// 1) transposing into a larger matrix by adding padding
// 2) transposing into a smaller matrix by optionally removing padding. This is the general version
//		without restrictions, see the 'transpose.opencl' file for a faster but more restricted
//		transpose kernel.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
//R"(
#include "../common.glsl"
#include "level3.glsl"
// =================================================================================================

// just define some shader parameters here, they are basically the same across all
#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = PADTRA_TILE, local_size_y = PADTRA_TILE, local_size_z = 1) in;
#endif

#if USE_BDA == 0
	#if 1
		layout(binding = 0, std430) buffer src_offsets_buf { int src_offsets[]; };
		layout(binding = 1, std430) buffer src_buf { real src[]; };
		layout(binding = 2, std430) buffer dest_offsets_buf { int dest_offsets[]; };
		layout(binding = 3, std430) buffer dest_buf { real dest[]; };
	#else
		layout(binding = 0, std430) buffer src_buf { real src[]; };
		layout(binding = 1, std430) buffer dest_buf { real dest[]; };
	#endif
#endif

shared real tile[(PADTRA_WPT*PADTRA_TILE) * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD)];


// Transposes a matrix from source to destination. The output is padded with zero values in case the
// destination matrix dimensions are larger than the transposed source matrix dimensions.
void _TransposePadMatrix(
#if USE_BDA
	LOCAL_PTR real* tile,
#endif
	const int src_one, const int src_two,
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
	// Loop over the work per thread
	// #pragma unroll
	for (int _w_one = 0; _w_one < PADTRA_WPT; _w_one += 1) {
		// #pragma unroll
		for (int _w_two = 0; _w_two < PADTRA_WPT; _w_two += 1) {

			// Computes the identifiers for the source matrix. Note that the local and global dimensions
			// do not correspond to each other!
			const int id_src_one = (get_group_id(1)*PADTRA_WPT + _w_two) * PADTRA_TILE + get_local_id(0);
			const int id_src_two = (get_group_id(0)*PADTRA_WPT + _w_one) * PADTRA_TILE + get_local_id(1);

			// Loads data into the local memory if the thread IDs are within bounds of the source matrix.
			// Otherwise, set the local memory value to zero.
			real value;
			SetToZero(value);
			if (id_src_two < src_two && id_src_one < src_one) {
				value = src[id_src_two*src_ld + id_src_one + src_offset];
			}
			const int tile_id0 = get_local_id(0)*PADTRA_WPT + _w_one;
			const int tile_id1 = get_local_id(1)*PADTRA_WPT + _w_two;
			tile[tile_id1 * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD) + tile_id0] = value;
		}
	}

	// Synchronizes all threads in a workgroup
	barrier();

	// Loop over the work per thread
	// #pragma unroll
	for (int _w_one = 0; _w_one < PADTRA_WPT; _w_one += 1) {
		// #pragma unroll
		for (int _w_two = 0; _w_two < PADTRA_WPT; _w_two += 1) {

			// Computes the identifiers for the destination matrix
			const int id_dest_one = (get_group_id(0)*PADTRA_WPT + _w_one) * PADTRA_TILE + get_local_id(0);
			const int id_dest_two = (get_group_id(1)*PADTRA_WPT + _w_two) * PADTRA_TILE + get_local_id(1);

			// Stores the transposed value in the destination matrix
			if ((id_dest_one < dest_one) && (id_dest_two < dest_two)) {
				const int tile_id0 = get_local_id(1)*PADTRA_WPT + _w_one;
				const int tile_id1 = get_local_id(0)*PADTRA_WPT + _w_two;
				real value = tile[tile_id1 * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD) + tile_id0];
				if (do_conjugate == 1) { COMPLEX_CONJUGATE(value); }
				Multiply(dest[id_dest_two*dest_ld + id_dest_one + dest_offset], alpha, value);
			}
		}
	}
}

// =================================================================================================

// End of the C++11 raw string literal
//)"

// =================================================================================================
