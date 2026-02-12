
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xger kernels for rank-1 matrix update.
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================

// Regular version of the rank-1 matrix update kernel (GER, GERU, GERC)
#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = WGS1, local_size_y = WGS2, local_size_z = 1) in;
	//__kernel __attribute__((reqd_work_group_size(WGS1, WGS2, 1)))
#endif

#if USE_BDA
#else
	layout(binding = 0, std430) buffer xgm_buf { real xgm[]; };
	layout(binding = 1, std430) buffer ygm_buf { real ygm[]; };
	layout(binding = 2, std430) buffer agm_buf { real agm[]; };
#endif

layout(push_constant) uniform Xger
{
	int max1;
	int max2;
	real_arg arg_alpha;
#if USE_BDA
	__global real* restrict xgm;
#endif
	int x_offset;
	int x_inc;
#if USE_BDA
	__global real* ygm;
#endif
	int y_offset;
	int y_inc;
#if USE_BDA
	__global real* restrict agm;
#endif
	int a_offset;
	int a_ld;
	int is_rowmajor;
} args;

void main()
{
	const real alpha = GetRealArg(args.arg_alpha);

	// Register storage for X and Y
	//#pragma promote_to_registers
	real xvalues[WPT];
	//#pragma promote_to_registers
	real yvalues[WPT];

	// Row-major version
	if (bool(args.is_rowmajor)) {

		// Loads the X-vector
		//#pragma unroll
		for (int _w = 0; _w < WPT; _w += 1) {
			const int id2 = _w*get_global_size(1) + get_global_id(1);
#if 1
			LoadVector(xvalues[_w], id2, args.max2, xgm, args.x_offset, args.x_inc, false);
#else
			xvalues[_w] = LoadVector(id2, args.max2, xgm, args.x_offset, args.x_inc, false);
#endif
		}

		// Loads the Y-vector
		//#pragma unroll
		for (int _w = 0; _w < WPT; _w += 1) {
			const int id1 = _w*get_global_size(0) + get_global_id(0);
#if 1
			LoadVector(yvalues[_w], id1, args.max1, ygm, args.y_offset, args.y_inc, true);
#else
			yvalues[_w] = LoadVector(id1, args.max1, ygm, args.y_offset, args.y_inc, true);
#endif
		}

		// Loops over the work per thread twice
		//#pragma unroll
		for (int _w1 = 0; _w1 < WPT; _w1 += 1) {
			//#pragma unroll
			for (int _w2 = 0; _w2 < WPT; _w2 += 1) {

				// Global thread IDs
				const int id1 = _w1*get_global_size(0) + get_global_id(0);
				const int id2 = _w2*get_global_size(1) + get_global_id(1);

				// Loads A, performs the operation, and stores the result into A
				MatrixUpdate(id1, id2, args.max1, args.max2, agm, args.a_offset, args.a_ld,
										 alpha, xvalues[_w2], yvalues[_w1], false);
			}
		}
	}

	// Col-major version
	else {

		// Loads the X-vector
		//#pragma unroll
		for (int _w = 0; _w < WPT; _w += 1) {
			const int id1 = _w*get_global_size(0) + get_global_id(0);
#if 1
			LoadVector(xvalues[_w], id1, args.max1, xgm, args.x_offset, args.x_inc, false);
#else
			xvalues[_w] = LoadVector(id1, args.max1, xgm, args.x_offset, args.x_inc, false);
#endif
		}

		// Loads the Y-vector
		//#pragma unroll
		for (int _w = 0; _w < WPT; _w += 1) {
			const int id2 = _w*get_global_size(1) + get_global_id(1);
#if 1
			LoadVector(yvalues[_w], id2, args.max2, ygm, args.y_offset, args.y_inc, true);
#else
			yvalues[_w] = LoadVector(id2, args.max2, ygm, args.y_offset, args.y_inc, true);
#endif
		}

		// Loops over the work per thread twice
		//#pragma unroll
		for (int _w1 = 0; _w1 < WPT; _w1 += 1) {
			//#pragma unroll
			for (int _w2 = 0; _w2 < WPT; _w2 += 1) {

				// Global thread IDs
				const int id1 = _w1*get_global_size(0) + get_global_id(0);
				const int id2 = _w2*get_global_size(1) + get_global_id(1);

				// Loads A, performs the operation, and stores the result into A
				MatrixUpdate(id1, id2, args.max1, args.max2, agm, args.a_offset, args.a_ld,
										 alpha, xvalues[_w1], yvalues[_w2], false);
			}
		}
	}
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
