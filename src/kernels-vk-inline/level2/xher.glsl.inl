// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xher kernels for rank-1 matrix update.
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================

// Symmetric version of the rank-1 matrix update kernel (HER, HPR, SYR, SPR)
#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = WGS1, local_size_y = WGS2, local_size_z = 1) in;
#endif

#if USE_BDA == 0
	layout(binding = 0, std430) buffer xgm_buf { real xgm[]; };
	layout(binding = 1, std430) buffer agm_buf { real agm[]; };
#endif

layout(push_constant) uniform Xher
{
	int n;
	real_arg arg_alpha;
#if USE_BDA
	__global real* restrict xgm;
#endif
	int x_offset; int x_inc;
#if USE_BDA
	__global real* restrict agm;
#endif
	int a_offset; int a_ld;
	int is_upper; int is_rowmajor;
} args;

void main()
{
	const real alpha = GetRealArg(args.arg_alpha);
	const bool is_upper = bool(args.is_upper);
	const bool is_rowmajor = bool(args.is_rowmajor);

	// Register storage for X and XT
	//#pragma promote_to_registers
	real xvalues[WPT];
	//#pragma promote_to_registers
	real xtvalues[WPT];

	// Loads the X-vector
	//#pragma unroll
	for (int _w = 0; _w < WPT; _w += 1) {
		const int id2 = _w*get_global_size(1) + get_global_id(1);
		LoadVector(xvalues[_w], id2, args.n, xgm, args.x_offset, args.x_inc, !is_rowmajor);
	}

	// Loads the X-transposed-vector
	//#pragma unroll
	for (int _w = 0; _w < WPT; _w += 1) {
		const int id1 = _w*get_global_size(0) + get_global_id(0);
		LoadVector(xtvalues[_w], id1, args.n, xgm, args.x_offset, args.x_inc, is_rowmajor);
	}

	// Loops over the work per thread twice
	//#pragma unroll
	for (int _w1 = 0; _w1 < WPT; _w1 += 1) {
		//#pragma unroll
		for (int _w2 = 0; _w2 < WPT; _w2 += 1) {

			// Global thread IDs
			const int id1 = _w1*get_global_size(0) + get_global_id(0);
			const int id2 = _w2*get_global_size(1) + get_global_id(1);

			// Skip these threads if they do not contain threads contributing to the matrix-triangle
			if ((is_upper && (id1 > id2)) || (!is_upper && (id2 > id1))) {
				// Do nothing
			}

			// Loads A, performs the operation, and stores the result into A
			else {
				MatrixUpdate(id1, id2, args.n, args.n, agm, args.a_offset, args.a_ld, alpha, xvalues[_w2], xtvalues[_w1], is_upper);
			}
		}
	}
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
