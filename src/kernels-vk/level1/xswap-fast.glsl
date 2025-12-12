
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

// =================================================================================================

// =================================================================================================

// Faster version of the kernel without offsets and strided accesses. Also assumes that 'n' is
// dividable by 'VW', 'WGS' and 'WPT'.


#if RELAX_WORKGROUP_SIZE == 0
	//__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
	layout(local_size_x = WGS, local_size_y = 1, local_size_z = 1) in;
#endif

#if USE_BDA == 0
	layout(binding = 0, std430) buffer xgm_buf { realV xgm[]; };
	layout(binding = 1, std430) buffer ygm_buf { realV ygm[]; };
#endif

layout(push_constant) uniform XswapFast
{
	int n;
#if USE_BDA
	__global realV* xgm;
	__global realV* ygm;
#endif
} args;

//void XswapFast()
void main()
{
	if ( !(args.n % VW == 0 && args.n % WPT == 0 && args.n % WGS == 0) ) return;
	for (int _w = 0; _w < WPT; _w += 1)
	{
		const int id = _w*get_global_size(0) + get_global_id(0);
		realV temp = xgm[id];
		xgm[id] = ygm[id];
		ygm[id] = temp;
	}
}
// =================================================================================================

// End of the C++11 raw string literal
//)"

// =================================================================================================
