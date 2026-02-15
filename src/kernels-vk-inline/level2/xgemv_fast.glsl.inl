
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xgemv kernel (fast versions) for matrix-vector multiplication.
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
R"(
// =================================================================================================

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library.

// 1: For the full version, see 'xgemv.opencl'

// 2: For the fast version
#ifndef WGS2
	#define WGS2 64		 // The local work-group size
#endif
#ifndef WPT2
	#define WPT2 1			// The amount of work-per-thread
#endif
#ifndef VW2
	#define VW2 1			 // Vector width of matrix A loads
#endif

// =================================================================================================

// Data-widths for the 'fast' kernel
#if VW2 == 1
	#define realVF real
#elif VW2 == 2
	#define realVF real2
#elif VW2 == 4
	#define realVF real4
#elif VW2 == 8
	#define realVF real8
#elif VW2 == 16
	#define realVF real16
#endif

// buffer declarations
#if USE_BDA == 0
	layout(binding = 0, std430) buffer agm_buf { realVF agm[]; };
	layout(binding = 1, std430) buffer xgm_buf { real xgm[]; };
	layout(binding = 2, std430) buffer ygm_buf { real ygm[]; };
#endif

// =================================================================================================

// Faster version of the kernel, assuming that:
// --> 'm' and 'n' are multiples of WGS2
// --> 'a_offset' is 0
// --> 'a_ld' is a multiple of VW2
// --> 'a_rotated' is 0
// --> 'do_conjugate' is 0
#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = WGS2, local_size_y = 1, local_size_z = 1) in;
#endif

layout(push_constant) uniform XgemvFast
{
	int m;
	int n;
	real_arg arg_alpha;
	real_arg arg_beta;
	//int a_rotated;
#if USE_BDA
	__global realVF* restrict agm;
#endif
	//int a_offset;
	int a_ld;
#if USE_BDA
	__global real* restrict xgm;
#endif
	int x_offset;
	int x_inc;
#if USE_BDA
	__global real* ygm;
#endif
	int y_offset; int y_inc;
	//int do_conjugate;
	//int parameter;
	//int kl_unused;
	//int ku_unused;
} args;

// Local memory for the vector X
shared real xlm[WGS2];

void main()
{
	if (!(args.m % WGS2 == 0 && args.n % WGS2 == 0 && args.a_ld % VW2 == 0)) return;
	
	
	const real alpha = GetRealArg(args.arg_alpha);
	const real beta = GetRealArg(args.arg_beta);

	// Local memory for the vector X
	//__local real xlm[WGS2];

	// Initializes the accumulation registers
	// #pragma promote_to_registers
	real acc2[WPT2];
	
	for (int _w = 0; _w < WPT2; _w += 1)
	{
		SetToZero(acc2[_w]);
	}

	// Loops over work-group sized portions of the work
	for (int kwg=0; kwg<args.n; kwg+=WGS2) {

		// Loads the vector X into local memory
		const int lid = get_local_id(0);
		xlm[lid] = xgm[(kwg + lid)*args.x_inc + args.x_offset];

		// Synchronizes all threads in a workgroup
		barrier();

		// The multiply-add function (not rotated)
		
		for (int _kl = 0; _kl < WGS2; _kl += 1) {
			const int k = kwg + _kl;
			
			for (int _w = 0; _w < WPT2/VW2; _w += 1) {
				const int gid = (WPT2/VW2)*get_global_id(0) + _w;
				realVF avec = agm[(args.a_ld/VW2)*k + gid];
				#if VW2 == 1
					MultiplyAdd(acc2[VW2*_w+0], xlm[_kl], avec);
				#else
					UNROLL(VW2)
					for (uint iv = 0; iv < VW2; iv += 1)
						MultiplyAdd(acc2[VW2*_w+iv], xlm[_kl], avec.s[iv]);
				#endif
			}
		}

		// Synchronizes all threads in a workgroup
		barrier();
	}

	// Stores the final result
	
	for (int _w = 0; _w < WPT2; _w += 1) {
		const int gid = WPT2*get_global_id(0) + _w;
		real yval = ygm[gid*args.y_inc + args.y_offset];
		AXPBY(ygm[gid*args.y_inc + args.y_offset], alpha, acc2[_w], beta, yval);
	}
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
