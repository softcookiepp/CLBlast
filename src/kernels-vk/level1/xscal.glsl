
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

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
//R"(

// =================================================================================================

// Full version of the kernel with offsets and strided accesses
#if RELAX_WORKGROUP_SIZE == 0
	//__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
	layout(local_size_x = WGS, local_size_y = 1, local_size_z = 1) in;
#endif

#if USE_BDA == 0
	layout(binding = 0, std430) buffer xgm_buf { real xgm[]; };
#endif

layout(push_constant) Xscal
{
	int n;
	real_arg arg_alpha;
#if USE_BDA
	__global real* xgm;
#endif
	int x_offset;
	int x_inc;
} args;

// Xscal
void main()
{
	const real alpha = GetRealArg(arg_alpha);

	// Loops over the work that needs to be done (allows for an arbitrary number of threads)
	for (int id = gl_GlobalInvocationID[0]; id<n; id += GET_GLOBAL_SIZE(0))
	{
		real xvalue = INDEX(xgm, id*x_inc + x_offset);
		real result;
		Multiply(result, alpha, xvalue);
		INDEX(xgm, id*x_inc + x_offset) = result;
	}
}

// =================================================================================================
#if 0
// Faster version of the kernel without offsets and strided accesses. Also assumes that 'n' is
// dividable by 'VW', 'WGS' and 'WPT'.
#if RELAX_WORKGROUP_SIZE == 1
	//__kernel
#else
	//__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
	layout(local_size_x = WGS, local_size_y = 1, local_size_z = 1) in;
#endif
void XscalFast(const int n, const real_arg arg_alpha,
							 __global realV* xgm) {
#if 0 //__has_builtin(__builtin_assume)
	__builtin_assume(n % VW == 0);
	__builtin_assume(n % WPT == 0);
	__builtin_assume(n % WGS == 0);
#endif
	const real alpha = GetRealArg(arg_alpha);

	#pragma unroll
	for (int _w = 0; _w < WPT; _w += 1) {
		const int id = _w*get_global_size(0) + get_global_id(0);
		realV xvalue = xgm[id];
		realV result;
		result = MultiplyVector(result, alpha, xvalue);
		xgm[id] = result;
	}
}
#endif
// =================================================================================================

// End of the C++11 raw string literal
//)"

// =================================================================================================
