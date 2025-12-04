#version 450
#include "../common.glsl"
#define ROUTINE_TRSV 1
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains kernels to perform forward or backward substition, as used in the TRSV routine
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
//R"(

// =================================================================================================
#if defined(ROUTINE_TRSV)

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library.

#ifndef TRSV_BLOCK_SIZE
	#define TRSV_BLOCK_SIZE 32		// The block size for forward or backward substition
#endif

// buffers
#if USE_BDA == 0
	layout(binding = 0, std430) buffer A_buf { real A[]; };
	layout(binding = 1, std430) buffer b_buf { real b[]; };
	layout(binding = 2, std430) buffer x_buf { real x[]; };
#endif

// =================================================================================================

#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = TRSV_BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;
#endif

void trsv_forward(int n,
									const __global real *A, const int a_offset, int a_ld,
									__global real *b, const int b_offset, int b_inc,
									__global real *x, const int x_offset, int x_inc,
									const int is_transposed, const int is_unit_diagonal, const int do_conjugate) {
	__local real alm[TRSV_BLOCK_SIZE][TRSV_BLOCK_SIZE];
	__local real xlm[TRSV_BLOCK_SIZE];
	const int tid = get_local_id(0);

	// Pre-loads the data into local memory
	if (tid < n) {
		Subtract(xlm[tid], b[tid*b_inc + b_offset], x[tid*x_inc + x_offset]);
		if (is_transposed == 0) {
			for (int i = 0; i < n; ++i) {
				alm[i][tid] = A[i + tid*a_ld + a_offset];
			}
		}
		else {
			for (int i = 0; i < n; ++i) {
				alm[i][tid] = A[tid + i*a_ld + a_offset];
			}
		}
		if (do_conjugate) {
			for (int i = 0; i < n; ++i) {
				COMPLEX_CONJUGATE(alm[i][tid]);
			}
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// Computes the result (single-threaded for now)
	if (tid == 0) {
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < i; ++j) {
				MultiplySubtract(xlm[i], alm[i][j], xlm[j]);
			}
			if (is_unit_diagonal == 0) { DivideFull(xlm[i], xlm[i], alm[i][i]); }
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// Stores the results
	if (tid < n) {
		x[tid*x_inc + x_offset] = xlm[tid];
	}
}


void trsv_backward(int n,
									 const __global real *A, const int a_offset, int a_ld,
									 __global real *b, const int b_offset, int b_inc,
									 __global real *x, const int x_offset, int x_inc,
									 const int is_transposed, const int is_unit_diagonal, const int do_conjugate) {
	__local real alm[TRSV_BLOCK_SIZE][TRSV_BLOCK_SIZE];
	__local real xlm[TRSV_BLOCK_SIZE];
	const int tid = get_local_id(0);

	// Pre-loads the data into local memory
	if (tid < n) {
		Subtract(xlm[tid], b[tid*b_inc + b_offset], x[tid*x_inc + x_offset]);
		if (is_transposed == 0) {
			for (int i = 0; i < n; ++i) {
				alm[i][tid] = A[i + tid*a_ld + a_offset];
			}
		}
		else {
			for (int i = 0; i < n; ++i) {
				alm[i][tid] = A[tid + i*a_ld + a_offset];
			}
		}
		if (do_conjugate) {
			for (int i = 0; i < n; ++i) {
				COMPLEX_CONJUGATE(alm[i][tid]);
			}
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// Computes the result (single-threaded for now)
	if (tid == 0) {
		for (int i = n - 1; i >= 0; --i) {
			for (int j = i + 1; j < n; ++j) {
				MultiplySubtract(xlm[i], alm[i][j], xlm[j]);
			}
			if (is_unit_diagonal == 0) { DivideFull(xlm[i], xlm[i], alm[i][i]); }
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// Stores the results
	if (tid < n) {
		x[tid*x_inc + x_offset] = xlm[tid];
	}
}

#endif
// =================================================================================================

// End of the C++11 raw string literal
//)"

// =================================================================================================
