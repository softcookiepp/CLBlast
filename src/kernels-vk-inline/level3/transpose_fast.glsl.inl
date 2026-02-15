

// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the common kernels shared among different BLAS functions. This file contains
// a kernel to transpose matrices. This is a 'fast' version with restrictions, see the
// 'padtranspose.opencl' file for a general transpose kernel.
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
R"(
// =================================================================================================

// Data-widths
#if TRA_WPT == 1
	#define realT real
#elif TRA_WPT == 2
	#define realT real2
#elif TRA_WPT == 4
	#define realT real4
#elif TRA_WPT == 8
	#define realT real8
#elif TRA_WPT == 16
	#define realT real16
#endif

// =================================================================================================

// Transposes and copies a matrix. Requires both matrices to be of the same dimensions and without
// offset. A more general version is available in 'padtranspose.opencl'.
#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = TRA_DIM, local_size_y = TRA_DIM, local_size_z = 1) in;
#endif

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer src_buf { realT src[]; };
	layout(binding = 1, std430) writeonly buffer dest_buf { realT dest[]; };
#endif

layout(push_constant) uniform TransposeMatrixFast
{
	int ld;
#if USE_BDA
	__global realT* restrict src,
	__global realT* dest,
#endif
	real_arg arg_alpha;
} args;

// Local memory to store a tile of the matrix (for coalescing)
shared realT tile[TRA_WPT*TRA_DIM][TRA_DIM + TRA_PAD];

void main()
{
	const real alpha = GetRealArg(args.arg_alpha);

	// Sets the group identifiers. They might be 'shuffled' around to distribute work in a different
	// way over workgroups, breaking memory-bank dependencies.
	const int gid0 = get_group_id(0);
	#if TRA_SHUFFLE == 1
		const int gid1 = (get_group_id(0) + get_group_id(1)) % get_num_groups(0);
	#else
		const int gid1 = get_group_id(1);
	#endif

	// Loops over the work per thread
	// #pragma unroll
	for (int _w_one = 0; _w_one < TRA_WPT; _w_one += 1) {

		// Computes the identifiers for the source matrix. Note that the local and global dimensions
		// do not correspond to each other!
		const int id_one = gid1 * TRA_DIM + get_local_id(0);
		const int id_two = (gid0 * TRA_DIM + get_local_id(1))*TRA_WPT + _w_one;

		// Loads data into the local memory
		realT value = src[id_two*(args.ld/TRA_WPT) + id_one];
		tile[get_local_id(0)*TRA_WPT + _w_one][get_local_id(1)] = value;
	}

	// Synchronizes all threads in a workgroup
	barrier();

	// Loads transposed data from the local memory
	// #pragma promote_to_registers
	realT vpm[TRA_WPT];
	// #pragma unroll
	for (int _w_one = 0; _w_one < TRA_WPT; _w_one += 1) {
		vpm[_w_one] = tile[get_local_id(1)*TRA_WPT + _w_one][get_local_id(0)];
	}

	// Performs the register-level transpose of the vectorized data
	// #pragma promote_to_registers
	realT results[TRA_WPT];
	#if TRA_WPT == 1
		results[0] = vpm[0];
	#else
		vTranspose(results, vpm, TRA_WPT);
	#endif

	// Multiplies by alpha and then stores the results into the destination matrix
	UNROLL(TRA_WPT)
	for (int _w_two = 0; _w_two < TRA_WPT; _w_two += 1) {
		realT result;
		#if TRA_WPT == 1
			Multiply(result, alpha, results[_w_two]);
		#else
			vsMultiply(result, alpha, results[_w_two], TRA_WPT);
		#endif
		const int id_one = gid0*TRA_DIM + get_local_id(0);
		const int id_two = (gid1*TRA_DIM + get_local_id(1))*TRA_WPT + _w_two;
		dest[id_two*(args.ld/TRA_WPT) + id_one] = result;
	}
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
