#version 450
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xswap kernel. It contains one fast vectorized version in case of unit
// strides (incx=incy=1) and no offsets (offx=offy=0). Another version is more general, but doesn't
// support vector data-types.
//
// This kernel uses the level-1 BLAS common tuning parameters.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
//R"(
#include "../common.glsl"
#include "level1.glsl"
// =================================================================================================

// Full version of the kernel with offsets and strided accesses
#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = WGS, local_size_y = 1, local_size_z = 1) in;
#endif

#if USE_BDA == 0
	layout(binding = 0, std430) buffer xgm_buf { real xgm[]; };
	layout(binding = 1, std430) buffer ygm_buf { real ygm[]; };
#endif

layout(push_constant, std430) uniform Xswap
{
	int n;
#if USE_BDA
	real_ptr_t xgm;
#endif
	int x_offset;
	int x_inc;
#if USE_BDA
	real_ptr_t ygm;
#endif
	int y_offset;
	int y_inc;
};

//void Xswap(
void main()
{
	// Loops over the work that needs to be done (allows for an arbitrary number of threads)
	for (int id = get_global_id(0); id<n; id += get_global_size(0))
	{
		real temp = indexGM(xgm, id*x_inc + x_offset);
		indexGM(xgm, id*x_inc + x_offset) = indexGM(ygm, id*y_inc + y_offset);
		indexGM(ygm, id*y_inc + y_offset) = temp;
	}
}

// =================================================================================================

// End of the C++11 raw string literal
//)"

// =================================================================================================
