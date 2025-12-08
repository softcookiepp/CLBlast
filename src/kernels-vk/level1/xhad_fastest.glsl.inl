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
	#elif VW == 2
		Multiply(cvec.x, aval.x, bvec.x);
		Multiply(cvec.y, aval.y, bvec.y);
	#elif VW == 4
		Multiply(cvec.x, aval.x, bvec.x);
		Multiply(cvec.y, aval.y, bvec.y);
		Multiply(cvec.z, aval.z, bvec.z);
		Multiply(cvec.w, aval.w, bvec.w);
	#elif VW == 8
#if PRECISION == 16 || PRECISION  == 32 || PRECISION == 64
		Multiply(cvec[0], aval[0], bvec[0]);
		Multiply(cvec[1], aval[1], bvec[1]);
#else
		Multiply(cvec.s0, aval.s0, bvec.s0);
		Multiply(cvec.s1, aval.s1, bvec.s1);
		Multiply(cvec.s2, aval.s2, bvec.s2);
		Multiply(cvec.s3, aval.s3, bvec.s3);
		Multiply(cvec.s4, aval.s4, bvec.s4);
		Multiply(cvec.s5, aval.s5, bvec.s5);
		Multiply(cvec.s6, aval.s6, bvec.s6);
		Multiply(cvec.s7, aval.s7, bvec.s7);
#endif
	#elif VW == 16
#if PRECISION == 16 || PRECISION  == 32 || PRECISION == 64
		Multiply(cvec[0], aval[0], bvec[0]);
		Multiply(cvec[1], aval[1], bvec[1]);
		Multiply(cvec[2], aval[2], bvec[2]);
		Multiply(cvec[3], aval[3], bvec[3]);
#else
		Multiply(cvec.s0, aval.s0, bvec.s0);
		Multiply(cvec.s1, aval.s1, bvec.s1);
		Multiply(cvec.s2, aval.s2, bvec.s2);
		Multiply(cvec.s3, aval.s3, bvec.s3);
		Multiply(cvec.s4, aval.s4, bvec.s4);
		Multiply(cvec.s5, aval.s5, bvec.s5);
		Multiply(cvec.s6, aval.s6, bvec.s6);
		Multiply(cvec.s7, aval.s7, bvec.s7);
		Multiply(cvec.s8, aval.s8, bvec.s8);
		Multiply(cvec.s9, aval.s9, bvec.s9);
		Multiply(cvec.sA, aval.sA, bvec.sA);
		Multiply(cvec.sB, aval.sB, bvec.sB);
		Multiply(cvec.sC, aval.sC, bvec.sC);
		Multiply(cvec.sD, aval.sD, bvec.sD);
		Multiply(cvec.sE, aval.sE, bvec.sE);
		Multiply(cvec.sF, aval.sF, bvec.sF);
#endif
	#endif
	return cvec;
}

// =================================================================================================

#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = WGS, local_size_y = 1, local_size_z = 1) in;
#endif

#if USE_BDA == 0
	layout(binding = 0, std430) buffer xgm_buf { realV xgm[]; };
	layout(binding = 1, std430) buffer ygm_buf { realV ygm[]; };
	layout(binding = 2, std430) buffer zgm_buf { realV zgm[]; };
#endif

layout(push_constant) uniform XhadFaster
{
	int n; real_arg arg_alpha; real_arg arg_beta;
#if USE_BDA
	__global real* restrict xgm;
	__global real* restrict ygm;
	__global real* zgm;
#endif
} args;

// Faster version of the kernel without offsets and strided accesses. Also assumes that 'n' is
// dividable by 'VW', 'WGS' and 'WPT'.

void main()
{
	if (!(args.n % VW == 0 && args.n % WPT == 0 && args.n % WGS == 0) ) return;
	
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

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
