
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xhad kernel. It contains one fast vectorized version in case of unit
// strides (incx=incy=incz=1) and no offsets (offx=offy=offz=0). Another version is more general,
// but doesn't support vector data-types. Based on the XAXPY kernels.
//
// This kernel uses the level-1 BLAS common tuning parameters.
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
R"(
// =================================================================================================

// A vector-vector multiply function. See also level1.opencl for a vector-scalar version
realV MultiplyVectorVector(realV cvec, const realV aval, const realV bvec) {
	#if VW == 1
		Multiply(cvec, aval, bvec);
	#else
		vMultiply(cvec, aval, bvec, VW);
	#endif
	return cvec;
}

// =================================================================================================

// Full version of the kernel with offsets and strided accesses
#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = WGS, local_size_y = 1, local_size_z = 1) in;
#endif

#if USE_BDA == 0
	layout(binding = 0, std430) buffer xgm_buf { real xgm[]; };
	layout(binding = 1, std430) buffer ygm_buf { real ygm[]; };
	layout(binding = 2, std430) buffer zgm_buf { real zgm[]; };
#endif

layout(push_constant) uniform Xhad
{
	int n; real_arg arg_alpha; real_arg arg_beta;
#if USE_BDA
	__global real* restrict xgm;
#endif
	int x_offset; int x_inc;
#if USE_BDA
	__global real* restrict ygm;
#endif
	int y_offset; int y_inc;
#if USE_BDA
	__global real* zgm;
#endif
	int z_offset; int z_inc;
} args;

void main()
{
	const real alpha = GetRealArg(args.arg_alpha);
	const real beta = GetRealArg(args.arg_beta);
	
	// Loops over the work that needs to be done (allows for an arbitrary number of threads)
	for (int id = get_global_id(0); id < args.n; id += get_global_size(0)) {
		real xvalue = xgm[id*args.x_inc + args.x_offset];
		real yvalue = ygm[id*args.y_inc + args.y_offset];
		real zvalue = zgm[id*args.z_inc + args.z_offset];
		real result;
		real alpha_times_x;
		Multiply(alpha_times_x, alpha, xvalue);
		Multiply(result, alpha_times_x, yvalue);
		MultiplyAdd(result, beta, zvalue);
		zgm[id*args.z_inc + args.z_offset] = result;
	}
}
#if 0
// Faster version of the kernel without offsets and strided accesses but with if-statement. Also
// assumes that 'n' is dividable by 'VW' and 'WPT'.
#if RELAX_WORKGROUP_SIZE == 1
	__kernel
#else
	__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
#endif
void XhadFaster(const int n, const real_arg arg_alpha, const real_arg arg_beta,
								const __global realV* restrict xgm, const __global realV* restrict ygm,
								__global realV* zgm) {
#if __has_builtin(__builtin_assume)
	__builtin_assume(n % VW == 0);
	__builtin_assume(n % WPT == 0);
#endif
	const real alpha = GetRealArg(args.arg_alpha);
	const real beta = GetRealArg(args.arg_beta);

	const int num_desired_threads = args.n / (VW * WPT);

	if (get_global_id(0) < num_desired_threads) {
		//#pragma unroll
		for (int _w = 0; _w < WPT; _w += 1) {
			const int id = _w * num_desired_threads + get_global_id(0);
			realV xvalue = xgm[id];
			realV yvalue = ygm[id];
			realV zvalue = zgm[id];
			realV result;
			realV alpha_times_x;
			alpha_times_x = MultiplyVector(alpha_times_x, alpha, xvalue);
			result = MultiplyVectorVector(result, alpha_times_x, yvalue);
			zgm[id] = MultiplyAddVector(result, beta, zvalue);
		}
	}
}
#endif
#if 0
// Faster version of the kernel without offsets and strided accesses. Also assumes that 'n' is
// dividable by 'VW', 'WGS' and 'WPT'.
#if RELAX_WORKGROUP_SIZE == 1
	__kernel
#else
	__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
#endif
void XhadFastest(const int n, const real_arg arg_alpha, const real_arg arg_beta,
								 const __global realV* restrict xgm, const __global realV* restrict ygm,
								 __global realV* zgm) {
#if __has_builtin(__builtin_assume)
	__builtin_assume(n % VW == 0);
	__builtin_assume(n % WPT == 0);
	__builtin_assume(n % WGS == 0);
#endif
	const real alpha = GetRealArg(args.arg_alpha);
	const real beta = GetRealArg(args.arg_beta);

	//#pragma unroll
	for (int _w = 0; _w < WPT; _w += 1) {
		const int id = _w*get_global_size(0) + get_global_id(0);
		realV xvalue = xgm[id];
		realV yvalue = ygm[id];
		realV zvalue = zgm[id];
		realV result;
		realV alpha_times_x;
		alpha_times_x = MultiplyVector(alpha_times_x, alpha, xvalue);
		result = MultiplyVectorVector(result, alpha_times_x, yvalue);
		zgm[id] = MultiplyAddVector(result, beta, zvalue);
	}
}
#endif

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
