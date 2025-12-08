// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xher2 kernels for rank-2 matrix update.
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================

// Symmetric version of the rank-2 matrix update kernel (HER2, HPR2, SYR2, SPR2)
#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = WGS1, local_size_y = WGS2, local_size_Z = 1) in;
#endif

#if USE_BDA == 0
	layout(binding = 0, std430) buffer xgm_buf { real xgm[]; };
	layout(binding = 1, std430) buffer ygm_buf { real ygm[]; };
	layout(binding = 2, std430) buffer agm_buf { real agm[]; };
#endif

layout(push_constant) uniform Xher2
{
	int n;
	real_arg arg_alpha;
#if USE_BDA
	__global real* restrict xgm;
#endif
	int x_offset; int x_inc;
#if USE_BDA
	__global real* restrict ygm;
#endif
	int y_offset; int y_inc;
#if USE_BDA
	__global real* restrict agm;
#endif
	int a_offset; int a_ld;
	int is_upper; int is_rowmajor;
} args;

void main()
{
	const real alpha = GetRealArg(args.arg_alpha);

	// Register storage for X and Y
	//#pragma promote_to_registers
	real xvalues[WPT];
	//#pragma promote_to_registers
	real yvalues[WPT];
	//#pragma promote_to_registers
	real xtvalues[WPT];
	//#pragma promote_to_registers
	real ytvalues[WPT];

	// Loads the X-vector
	//#pragma unroll
	for (int _w = 0; _w < WPT; _w += 1) {
		const int id2 = _w*get_global_size(1) + get_global_id(1);
		LoadVector(xvalues[_w], id2, args.n, xgm, args.x_offset, args.x_inc, !bool(args.is_rowmajor));
	}

	// Loads the X-transposed-vector
	//#pragma unroll
	for (int _w = 0; _w < WPT; _w += 1) {
		const int id1 = _w*get_global_size(0) + get_global_id(0);
		LoadVector(xtvalues[_w], id1, args.n, xgm, args.x_offset, args.x_inc, bool(args.is_rowmajor));
	}

	// Loads the Y-vector
	//#pragma unroll
	for (int _w = 0; _w < WPT; _w += 1) {
		const int id1 = _w*get_global_size(0) + get_global_id(0);
		LoadVector(yvalues[_w], id1, args.n, ygm, args.y_offset, args.y_inc, bool(args.is_rowmajor));
	}

	// Loads the Y-transposed-vector
	//#pragma unroll
	for (int _w = 0; _w < WPT; _w += 1) {
		const int id2 = _w*get_global_size(1) + get_global_id(1);
		LoadVector(ytvalues[_w], id2, args.n, ygm, args.y_offset, args.y_inc, !bool(args.is_rowmajor));
	}

	// Sets the proper value of alpha in case conjugation is needed
	real alpha1 = alpha;
	real alpha2 = alpha;
	#if defined(ROUTINE_HER2) || defined(ROUTINE_HPR2)
		if (bool(args.is_rowmajor)) {
			COMPLEX_CONJUGATE(alpha1);
		}
		else {
			COMPLEX_CONJUGATE(alpha2);
		}
	#endif

	// Loops over the work per thread twice
	//#pragma unroll
	for (int _w1 = 0; _w1 < WPT; _w1 += 1) {
		//#pragma unroll
		for (int _w2 = 0; _w2 < WPT; _w2 += 1) {

			// Global thread IDs
			const int id1 = _w1*get_global_size(0) + get_global_id(0);
			const int id2 = _w2*get_global_size(1) + get_global_id(1);

			// Skip these threads if they do not contain threads contributing to the matrix-triangle
			if ((bool(args.is_upper) && (id1 > id2)) || (!bool(args.is_upper) && (id2 > id1))) {
				// Do nothing
			}

			// Loads A, performs the operation, and stores the result into A
			else {
				MatrixUpdate2(id1, id2, args.n, args.n, agm, args.a_offset, args.a_ld,
											alpha1, xvalues[_w2], yvalues[_w1],
											alpha2, xtvalues[_w1], ytvalues[_w2], bool(args.is_upper));
			}
		}
	}
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
