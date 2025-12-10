// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains kernels to convert triangular matrices to/from general matrices.
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================
#if 1 //defined(ROUTINE_TRMM)

// Same as above, but now the matrix' data is stored in the upper-triangle
#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = PAD_DIMX, local_size_y = PAD_DIMY, local_size_z = 1) in;
#endif

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer src_buf { real src[]; };
	layout(binding = 0, std430) writeonly buffer dest_buf { real dest[]; };
#endif

layout(push_constant) uniform TriaUpperToSquared
{
	int src_dim;
	int src_ld; int src_offset;
#if USE_BDA
	__global real* restrict src;
#endif
	int dest_dim;
	int dest_ld; int dest_offset;
#if USE_BDA
	__global real* dest
#endif
	int unit_diagonal;
} args;

void main()
{
	const bool unit_diagonal = bool(args.unit_diagonal);
	// Loops over the work per thread in both dimensions
	// #pragma unroll
	for (int _w_one = 0; _w_one < PAD_WPTX; _w_one += 1) {
		const int id_one = (get_group_id(0)*PAD_WPTX + _w_one) * PAD_DIMX + get_local_id(0);
		// #pragma unroll
		for (int _w_two = 0; _w_two < PAD_WPTY; _w_two += 1) {
			const int id_two = (get_group_id(1)*PAD_WPTY + _w_two) * PAD_DIMY + get_local_id(1);
			if (id_two < args.dest_dim && id_one < args.dest_dim) {

				// Loads data from the upper-triangular matrix
				real result;
				SetToZero(result);
				if (id_two < args.src_dim && id_one < args.src_dim) {
					if (id_one <= id_two) { result = src[id_two*args.src_ld + id_one + args.src_offset]; }
					if (id_one == id_two && unit_diagonal) { SetToOne(result); }
					// Else: result is zero
				}

				// Stores the result in the destination matrix
				dest[id_two*args.dest_ld + id_one + args.dest_offset] = result;
			}
		}
	}
}

#endif
// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
