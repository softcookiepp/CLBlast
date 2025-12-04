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

// 1: For the full version, see 'xgemv.glsl'

// 2: For the fast version see xgemv_fast.glsl

// 3: For the fast rotated version
#ifndef WGS3
	#define WGS3 64		 // The local work-group size
#endif
#ifndef WPT3
	#define WPT3 1			// The tile-size
#endif
#ifndef VW3
	#define VW3 1			 // Vector width of matrix A loads
#endif

// =================================================================================================

// Data-widths for the 'fast' kernel with rotated matrix
#if VW3 == 1
	#define realVFR real
#elif VW3 == 2
	#define realVFR real2
#elif VW3 == 4
	#define realVFR real4
#elif VW3 == 8
	#define realVFR real8
#elif VW3 == 16
	#define realVFR real16
#endif

#if USE_BDA == 0
	layout(binding = 0, std430) buffer agm_buf { realVFR agm[]; };
	layout(binding = 1, std430) buffer xgm_buf { real xgm[]; };
	layout(binding = 2, std430) buffer ygm_buf { real ygm[]; };
#endif

// =================================================================================================

// Faster version of the kernel, assuming that:
// --> 'm' and 'n' are multiples of WGS3
// --> 'a_offset' is 0
// --> 'a_ld' is a multiple of VW3
// --> 'a_rotated' is 1
// --> 'do_conjugate' is 0
#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = WGS3, local_size_y = 1, local_size_z = 1) in;
#endif

layout(push_constant) uniform XgemvFastRot
{
	//const int m;
	int n;
	real_arg arg_alpha;
	real_arg arg_beta;
	//int a_rotated;
#if USE_BDA
	__global realVFR* restrict agm,
#endif
	//int a_offset;
	int a_ld;
#if USE_BDA
	__global real* restrict xgm,
#endif
	int x_offset;
	int x_inc;
#if USE_BDA
	__global real* ygm, 
#endif
	int y_offset;
	int y_inc;
	//int do_conjugate;
	//int parameter;
	//int kl_unused;
	//int ku_unused;
} args;

shared real tile[WPT3][WGS3];
shared real xlm[WPT3];

void main()
{
	const real alpha = GetRealArg(args.arg_alpha);
	const real beta = GetRealArg(args.arg_beta);

	// Local memory to store a tile of the matrix (for coalescing)
	//__local real tile[WPT3][WGS3];
	const int lid = int(gl_LocalInvocationID[0]);
	const int lid_mod = lid % (WPT3/VW3);
	const int lid_div = lid / (WPT3/VW3);

	// Local memory for the vector X
	//__local real xlm[WPT3];

	// Initializes the accumulation register
	real acc3;
	SetToZero(acc3);

	// Loops over tile-sized portions of the work
	for (int kwg=0; kwg<args.n; kwg+=WPT3) {

		// Loads the vector X into local memory
		if (lid < WPT3) {
			xlm[lid] = xgm[(kwg + lid) * args.x_inc + args.x_offset];
		}

		// Loads the matrix A into local memory
		
		for (int _kl = 0; _kl < WPT3/VW3; _kl += 1) {
			const int x = (kwg/VW3) + lid_mod;
			const int y = int(gl_WorkGroupID[0]) * WGS3 + lid_div * (WPT3/VW3) + _kl;
			realVFR avec = agm[(args.a_ld/VW3) * y + x];
			#if VW3 == 1
				tile[_kl*VW3 + 0][lid] = avec;
			#elif VW3 == 2
				tile[_kl*VW3 + 0][lid] = avec.x;
				tile[_kl*VW3 + 1][lid] = avec.y;
			#elif VW3 == 4
				tile[_kl*VW3 + 0][lid] = avec.x;
				tile[_kl*VW3 + 1][lid] = avec.y;
				tile[_kl*VW3 + 2][lid] = avec.z;
				tile[_kl*VW3 + 3][lid] = avec.w;
			#elif VW3 == 8
				tile[_kl*VW3 + 0][lid] = avec.s0;
				tile[_kl*VW3 + 1][lid] = avec.s1;
				tile[_kl*VW3 + 2][lid] = avec.s2;
				tile[_kl*VW3 + 3][lid] = avec.s3;
				tile[_kl*VW3 + 4][lid] = avec.s4;
				tile[_kl*VW3 + 5][lid] = avec.s5;
				tile[_kl*VW3 + 6][lid] = avec.s6;
				tile[_kl*VW3 + 7][lid] = avec.s7;
			#elif VW3 == 16
				tile[_kl*VW3 + 0][lid] = avec.s0;
				tile[_kl*VW3 + 1][lid] = avec.s1;
				tile[_kl*VW3 + 2][lid] = avec.s2;
				tile[_kl*VW3 + 3][lid] = avec.s3;
				tile[_kl*VW3 + 4][lid] = avec.s4;
				tile[_kl*VW3 + 5][lid] = avec.s5;
				tile[_kl*VW3 + 6][lid] = avec.s6;
				tile[_kl*VW3 + 7][lid] = avec.s7;
				tile[_kl*VW3 + 8][lid] = avec.s8;
				tile[_kl*VW3 + 9][lid] = avec.s9;
				tile[_kl*VW3 + 10][lid] = avec.sA;
				tile[_kl*VW3 + 11][lid] = avec.sB;
				tile[_kl*VW3 + 12][lid] = avec.sC;
				tile[_kl*VW3 + 13][lid] = avec.sD;
				tile[_kl*VW3 + 14][lid] = avec.sE;
				tile[_kl*VW3 + 15][lid] = avec.sF;
			#endif
			barrier();
		}

		// Synchronizes all threads in a workgroup
		barrier();

		// The multiply-add function (rotated)
		
		for (int _kl = 0; _kl < WPT3/VW3; _kl += 1) {
			
			for (int _v = 0; _v < VW3; _v += 1) {
				real aval = tile[lid_mod*VW3 + _v][lid_div * (WPT3/VW3) + _kl];
				real xval = xlm[_kl*VW3 + _v];
				MultiplyAdd(acc3, xval, aval);
			}
		}

		// Synchronizes all threads in a workgroup
		barrier();
	}

	// Stores the final result
	const int gid = int(gl_GlobalInvocationID[0]);
	real yval = ygm[gid * args.y_inc + args.y_offset];
	AXPBY(ygm[gid * args.y_inc + args.y_offset], alpha, acc3, beta, yval);
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
