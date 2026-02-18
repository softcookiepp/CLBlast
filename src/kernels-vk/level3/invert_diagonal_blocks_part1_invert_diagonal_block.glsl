#version 450

// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains kernels to invert squared diagonal blocks of a matrix. These kernels are based
// on the TRSM implementation in the CUDA version of Magma version 2.2.0 and the poster "Triangular
// Linear System Solver for GPU with CUDA and OpenCL" by Peng Du, Stanimire Tomov, Piotr Luszczek,
// and Jack Dongarra.
//
// This is part 1 of 2, see part 2 for the remainder of the kernel code.
//
// =================================================================================================
//
//	Let A be an block_size*block_size lower triangular matrix, and B its inverse.
//	Then the block decomposition
//	
//			[ A11	 0	] * [ B11	 0	] = [ I 0 ]
//			[ A21	A22 ]	 [ B21	B22 ]	 [ 0 I ]
//	
//	yields
//	
//			A11*B11 = I						==>	B11 =	A11^{-1},
//			A22*B22 = I						==>	B22 =	A22^{-1},
//			A21*B11 + A22*B21 = 0	==>	B21 = -A22^{-1}*A21*B11 = -B22*A21*B11.
//	
//	The InvertDiagonalBlock kernel inverts A11 and A22.
//	The TripleMatMul routines multiply:
//	part 1:	B21 =	A21 * B11,
//	part 2:	B21 = -B22 * B21.
//	
//	At this level, inner block is current_size=16, with one 4 x 4 work-group per inner block. Each
//	submatrix Aij and Bij is current_size x current_size. The submatrix dimension is multiplied by 2
//	at each level, so the next level is current_size*2 = 32. A 'page' is the next bigger block,
//	here current_size*2=32,
//								 [ B11	 0	]
//	which contains [ B21	B22 ].
//	Outer blocks are block_size x block_size.
//	
//	A21 may have < current_size rows, but is guaranteed to have current_size cols since A22 is on
//	the right. This makes a single check easy to do.
//	
//	B is stored in workspace that is a full multiple of block_size x block_size; no checks needed.
//	
//	We split this into part1 & part2 to synchronize all blocks and make sure
//	that writes to B12 are observed by all blocks.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
//R"(
#include "../common.glsl"
#include "level3.glsl"
// =================================================================================================
#if 1//defined(ROUTINE_INVERT)

// Parameters set by the tuner
// TODO: Make these actually tunable
#ifndef INTERNAL_BLOCK_SIZE
	#define INTERNAL_BLOCK_SIZE 16		 // Internal block size of the invert kernel
#endif
#ifndef LOCALPAD
	#define LOCALPAD 0								 // Padding in the x-dimension of the local memory to avoid bank conflicts
#endif
#ifndef LOCALX
	#define LOCALX (16 + LOCALPAD)		 // Local memory size in x-dimension of TripleMatMul kernels
#endif
#ifndef LOCALY
	#define LOCALY 16									// Local memory size in y-dimension of TripleMatMul kernels
#endif
#ifndef TMMWGSX
	#define TMMWGSX 4									// Work-group size in x-dimension of TripleMatMul kernels
#endif
#ifndef TMMWGSY
	#define TMMWGSY 4									// Work-group size in y-dimension of TripleMatMul kernels
#endif

// =================================================================================================

// Inverts a diagonal block of INTERNAL_BLOCK_SIZE by INTERNAL_BLOCK_SIZE elements in a larger matrix
#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = INTERNAL_BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;
#endif

#if USE_BDA == 0
	layout(binding = 0, std430) buffer src_buf { real src[]; };
	layout(binding = 1, std430) buffer dest_buf { real dest[]; };
#endif

layout(push_constant, std430) uniform InvertDiagonalBlock
{
	int n;
#if USE_BDA
	__global real* restrict src;
#endif
	int src_offset; int src_ld;
#if USE_BDA
	__global real* restrict dest;
#endif
	int outer_block_size;
	int unit_diagonal; int is_upper;
};

// Local memory to store the inverted block of INTERNAL_BLOCK_SIZE by INTERNAL_BLOCK_SIZE
shared real lm[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];

void main()
{
	const int thread_index = get_local_id(0);
	const int block_index = get_group_id(0);

	// Sets the offset for this particular block in the source and destination matrices
	const int block_index_per_block = block_index * INTERNAL_BLOCK_SIZE;
	const int src_block_offset = block_index * (INTERNAL_BLOCK_SIZE + src_ld * INTERNAL_BLOCK_SIZE) + src_offset;
	const int num_inner_blocks = outer_block_size / INTERNAL_BLOCK_SIZE;
	const int block_index_div = block_index / num_inner_blocks;
	const int block_index_mod = block_index % num_inner_blocks;
	const int offset_part1 = block_index_div * outer_block_size * outer_block_size; // go to the block_index_div outer outer_block_size*outer_block_size block
	const int offset_part2 = block_index_mod * (outer_block_size*INTERNAL_BLOCK_SIZE + INTERNAL_BLOCK_SIZE); // then to the block_index_mod inner INTERNAL_BLOCK_SIZE*INTERNAL_BLOCK_SIZE block inside that
	const int dest_block_offset = offset_part1 + offset_part2;

	// Loads the source lower triangle into local memory. Any values in the upper triangle or
	// outside of the matrix are set to zero
	for (int _j = 0; _j < INTERNAL_BLOCK_SIZE; _j += 1) {
		bool condition = false;
		if (is_upper != 0) {
			condition = (thread_index <= _j) && (block_index_per_block + _j < n);
		}
		else {
			condition = (thread_index >= _j) && (block_index_per_block + thread_index < n);
		}
		if (condition) {
			const int src_index = _j*src_ld + thread_index + src_block_offset;
			lm[thread_index][_j] = src[src_index];
		}
		else {
			SetToZero(lm[thread_index][_j]);
		}
	}
	barrier();

	// Inverts the diagonal
	real inverted_diagonal;
	SetToOne(inverted_diagonal);
	if (unit_diagonal == 0) {
		const real diagonal_value = lm[thread_index][thread_index];
		if (!IsZero(diagonal_value)) { // Only for non-singular values and values inside the matrix
			real constant_one;
			SetToOne(constant_one);
			DivideFull(inverted_diagonal, constant_one, diagonal_value);
		}
	}
	lm[thread_index][thread_index] = inverted_diagonal;
	barrier();

	// Upper-triangular
	if (is_upper != 0) {

		// Computes the elements 0:j-1 of the j-th column
		for (int j = 1; j < INTERNAL_BLOCK_SIZE; ++j) {
			real sum;
			if (thread_index < j) {
				SetToZero(sum);
				for (int k = 0; k < j; ++k) {
					MultiplyAdd(sum, lm[thread_index][k], lm[k][j]);
				}
			}
			barrier();
			if (thread_index < j) {
				real diagonal_value = lm[j][j];
				Negate(diagonal_value);
				Multiply(lm[thread_index][j], diagonal_value, sum);
			}
			barrier();
		}
	}

	// Lower triangular
	else {

		// Computes the elements j+1:INTERNAL_BLOCK_SIZE-1 of the j-th column
		for (int j = INTERNAL_BLOCK_SIZE - 2; j >= 0; --j) {
			real sum;
			if (thread_index > j) {
				SetToZero(sum);
				for (int k = j + 1; k < INTERNAL_BLOCK_SIZE; ++k) {
					MultiplyAdd(sum, lm[thread_index][k], lm[k][j]);
				}
			}
			barrier();
			if (thread_index > j) {
				real diagonal_value = lm[j][j];
				Negate(diagonal_value);
				Multiply(lm[thread_index][j], diagonal_value, sum);
			}
			barrier();
		}
	}
	
	// Writes the result to global memory
	#pragma unroll
	for (int j = 0; j < INTERNAL_BLOCK_SIZE; j += 1) {
		dest[j*outer_block_size + thread_index + dest_block_offset] = lm[thread_index][j];
	}
}

#endif
// =================================================================================================

// End of the C++11 raw string literal
//)"

// =================================================================================================
