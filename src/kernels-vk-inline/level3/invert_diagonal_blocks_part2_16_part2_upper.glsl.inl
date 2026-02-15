

// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 2 of 2, see part 1 of the invert kernel for a description
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
R"(
// =================================================================================================

#if RELAX_WORKGROUP_SIZE == 0
	// local size appears to be variable for these kernels, so that is what we will do
	layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
#endif

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
} args;

shared real lm[LOCALY * LOCALX];

// B12 = -B11 * B12
//TripleMatMul16Part2Upper
void main()
{
	TripleMatMulPart2(16, true, lm, args.n, dest, args.current_size, args.num_pages, args.block_size);
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
