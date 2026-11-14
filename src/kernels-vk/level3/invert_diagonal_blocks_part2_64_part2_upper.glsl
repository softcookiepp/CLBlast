#version 450


// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 2 of 2, see part 1 of the invert kernel for a description
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
//R"(
#define INVERT_USE_SPEC 1
#include "invert_diagonal_blocks_part1_triple_matmul.glsl"
// =================================================================================================

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

#if USE_BDA == 0
	layout(binding = 0, std430) buffer dest_buf { real dest[]; };
#endif

layout(push_constant, std430) uniform TripleMatMulPart2
{
	int n;
#if USE_BDA
	__global real* restrict dest,
#endif
	int current_size; int num_pages; int block_size;
};

// B12 = -B11 * B12
//TripleMatMul64Part2Upper
void main()
{
	#if INVERT_USE_SPEC == 1
		TripleMatMulPart2(SIZE, UPPER, lm, n, dest, current_size, num_pages, block_size);
	#else
		TripleMatMulPart2(64, true, lm, n, dest, current_size, num_pages, block_size);
	#endif
}


// =================================================================================================

// End of the C++11 raw string literal
//)"

// =================================================================================================
