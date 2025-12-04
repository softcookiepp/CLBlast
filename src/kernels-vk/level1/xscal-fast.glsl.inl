
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xscal kernel. It contains one fast vectorized version in case of unit
// strides (incx=1) and no offsets (offx=0). Another version is more general, but doesn't support
// vector data-types.
//
// This kernel uses the level-1 BLAS common tuning parameters.
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================

// Faster version of the kernel without offsets and strided accesses. Also assumes that 'n' is
// dividable by 'VW', 'WGS' and 'WPT'.

#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = WGS, local_size_y = 1, local_size_z = 1) in;
#endif

#if USE_BDA == 0
	layout(binding = 0, std430) buffer xgm_buf { real xgm[]; };
#endif

layout(push_constant) XscalFast
{
	int n;
	real_arg arg_alpha;
#if USE_BDA
	__global real* xgm;
#endif
} args;

// XscalFast
void main()
{
	const real alpha = GetRealArg(arg_alpha);
	for (int _w = 0; _w < WPT; _w += 1)
	{
		const int id = _w*GET_GLOBAL_SIZE(0) + gl_GlobalInvocationID[0];
		realV xvalue = INDEX(xgm, id);
		realV result;
		result = MultiplyVector(result, alpha, xvalue);
		INDEX(xgm, id) = result;
	}
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
