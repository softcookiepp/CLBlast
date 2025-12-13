
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains kernels to perform forward or backward substition, as used in the TRSV routine
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================
#define ROUTINE_TRSV
#if defined(ROUTINE_TRSV)

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library.

#ifndef TRSV_BLOCK_SIZE
	#define TRSV_BLOCK_SIZE 32		// The block size for forward or backward substition
#endif

// buffers
#if USE_BDA == 0
	layout(binding = 0, std430) buffer A_buf { precise real A[]; };
	layout(binding = 1, std430) buffer b_buf { precise real b[]; };
	layout(binding = 2, std430) buffer x_buf { precise real x[]; };
#endif

// =================================================================================================

#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = TRSV_BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;
#endif

layout(push_constant) uniform trsv_forward
{
	int n;
#if USE_BDA
	__global real *A;
#endif
	int a_offset;
	int a_ld;
#if USE_BDA
	__global real *b;
#endif
	int b_offset;
	int b_inc;
#if USE_BDA
	__global real *x;
#endif
	int x_offset;
	int x_inc;
	int is_transposed;
	int is_unit_diagonal;
	int do_conjugate;
} args;

precise shared real alm[TRSV_BLOCK_SIZE][TRSV_BLOCK_SIZE];
precise shared real xlm[TRSV_BLOCK_SIZE];

void main()
{
	const int tid = get_local_id(0);

	// Pre-loads the data into local memory
	if (tid < args.n) {
		Subtract(xlm[tid], b[tid*args.b_inc + args.b_offset], x[tid*args.x_inc + args.x_offset]);
		if (args.is_transposed == 0) {
			for (int i = 0; i < args.n; ++i) {
				alm[i][tid] = A[i + tid*args.a_ld + args.a_offset];
			}
		}
		else {
			for (int i = 0; i < args.n; ++i) {
				alm[i][tid] = A[tid + i*args.a_ld + args.a_offset];
			}
		}
		if (args.do_conjugate != 0) {
			for (int i = 0; i < args.n; ++i) {
				COMPLEX_CONJUGATE(alm[i][tid]);
			}
		}
	}
	barrier();

	// Computes the result (single-threaded for now)
	if (tid == 0) {
		for (int i = 0; i < args.n; ++i) {
			for (int j = 0; j < i; ++j) {
				MultiplySubtract(xlm[i], alm[i][j], xlm[j]);
			}
			if (args.is_unit_diagonal == 0) { DivideFull(xlm[i], xlm[i], alm[i][i]); }
		}
	}
	barrier();

	// Stores the results
	if (tid < args.n) {
		x[tid*args.x_inc + args.x_offset] = xlm[tid];
	}
}

#endif
// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
