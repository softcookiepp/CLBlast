
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the batched version of the direct GEMM kernels. See part 1 for information
// about the non-batched version of the kernel.
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 3 of 3 of the GEMM kernel. See part 1 for more information.
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.

// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 2 of 3 of the GEMM kernel. See part 1 for more information.
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.

// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This is a generic GEMM kernel that works for all sizes and configurations: it doesn't require any
// pre and and post-processing kernels.
//
// This kernel is seperated into three files. This is part 1 out of 3.
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.

// literal). Comment-out this line for syntax-highlighting when developing.
#ifndef COMMON_GLSL
#define COMMON_GLSL
// =================================================================================================

#define USE_BDA 0

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this file is used outside of the CLBlast library.
#ifndef PRECISION
	#define PRECISION 32			// Data-types: half, single or double precision, complex or regular
#endif

// =================================================================================================
	
// reserved for when unrolling semantics are able to be used
#ifndef UNROLL
	#define UNROLL(N)
#endif

// Enable support for half-precision
#if PRECISION == 16
	#extension GL_EXT_shader_16bit_storage : require
	#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif

// Enable support for double-precision
#if PRECISION == 64 || PRECISION == 6464
	#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require
#endif


// Half-precision
#if PRECISION == 16
	struct vec2_t { f16vec2 s; };
	struct vec4_t { f16vec4 s; };
	struct vec8_t { float16_t s[8]; };
	struct vec16_t { float16_t s[16]; };
	#define real float16_t
	#define ZERO float16_t(0.0)
	#define ONE float16_t(1.0)
	#define SMALLEST -1.0e14

// Single-precision
#elif PRECISION == 32
	struct vec2_t { vec2 s; };
	struct vec4_t { vec4 s; };
	struct vec8_t { float s[8]; };
	struct vec16_t { float s[16]; };
	#define real float
	#define ZERO real(0.0f)
	#define ONE 1.0f
	#define SMALLEST -1.0e37f

// Double-precision 
#elif PRECISION == 64
	struct vec2_t { dvec2 s; };
	struct vec4_t { dvec4 s; };
	struct vec8_t { double s[8]; };
	struct vec16_t { double s[16]; };
	#define real double
	#define ZERO 0.0
	#define ONE 1.0
	#define SMALLEST -1.0e37

// Complex single-precision
#elif PRECISION == 3232
	struct vec2_t { mat2x2 s; };
	struct vec4_t { mat4x2 s; };
	struct vec8_t { vec2 s[8]; };
	struct vec16_t { vec2 s[16]; };
	#define real vec2
	#define ZERO 0.0f
	#define ONE 1.0f
	#define SMALLEST -1.0e37f

// Complex double-precision
#elif PRECISION == 6464
	struct vec2_t { dmat2x2 s; };
	struct vec4_t { dmat4x2 s; };					 
	struct vec8_t { dvec2 s[8]; };
	struct vec16_t { dvec2 s[16]; };
	#define real dvec2
	#define ZERO 0.0
	#define ONE 1.0
	#define SMALLEST -1.0e37
#endif

// this simplifies stuff c:
#define real2 vec2_t
#define real4 vec4_t
#define real8 vec8_t
#define real16 vec16_t

// Single-element version of a complex number
#if PRECISION == 3232
	#define singlereal float 
#elif PRECISION == 6464
	#define singlereal double 
#else
	#define singlereal real 
#endif

// Converts a 'real argument' value to a 'real' value as passed to the kernel. Normally there is no
// conversion, but half-precision is not supported as kernel argument so it is converted from float.
#if PRECISION == 16
	#define real_arg float 
	#define GetRealArg(x) float16_t(x)
#else
	#define real_arg real 
	#define GetRealArg(x) real(x)
#endif

//#elif PRECISION == 64
//	// lets see if this makes our life easier...
//	#define real_arg float
//	#define GetRealArg(x) double(x)

// Pointers to local memory objects (using a define because CUDA doesn't need them)
#ifndef LOCAL_PTR
	#define LOCAL_PTR shared
#endif

// =================================================================================================

// Don't use the non-IEEE754 compliant OpenCL built-in mad() instruction per default. For specific
// devices, this is enabled (see src/routine.cpp).
#ifndef USE_CL_MAD
	#define USE_CL_MAD 0
#endif

// By default the workgroup size requirement is enabled. For Qualcomm devices the workgroup size 
// requirement results in worse performance and is disabled (src/utilities/compile.cpp)
#ifndef RELAX_WORKGROUP_SIZE
	#define RELAX_WORKGROUP_SIZE 0
#endif

// ensure all spec constants related to workgroup size are here and ready
#if RELAX_WORKGROUP_SIZE
	layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
#endif

// Sets a variable to zero
#if PRECISION == 3232 || PRECISION == 6464
	#define SetToZero(a) a = real(ZERO, ZERO)
#else
	#define SetToZero(a) a = ZERO
#endif

// Sets a variable to zero (only the imaginary part)
#if PRECISION == 3232 || PRECISION == 6464
	#define ImagToZero(a) a.y = ZERO
#else
	#define ImagToZero(a) 
#endif

// Sets a variable to one
#if PRECISION == 3232 || PRECISION == 6464
	#define SetToOne(a) a = real(ONE, ZERO)
#else
	#define SetToOne(a) a = ONE
#endif

// Determines whether a variable is zero
#if PRECISION == 3232 || PRECISION == 6464
	#define IsZero(a) ((a[0] == ZERO) && (a[1] == ZERO))
#else
	#define IsZero(a) (a == ZERO)
#endif

// component-wise absolute value
#define AbsoluteValue(value) value = abs(value)

// Negation (component-wise)
#if 0 //PRECISION == 3232 || PRECISION == 6464
	#define Negate(value) value.x = (-1.0(value.x)); value.y = (-1.0*(value.y))
#else
	#define Negate(value) value = (-1.0*(value))
#endif

// Adds two complex variables
#if 0 //PRECISION == 3232 || PRECISION == 6464
	#define Add(c,a,b) c = real(a.x + b.x, a.y + b.y)
#else
	#define Add(c,a,b) c = a + b
#endif

// Subtracts two complex variables
#if 0 //PRECISION == 3232 || PRECISION == 6464
	#define Subtract(c,a,b) c = real(a.x - b.x, a.y - b.y)
#else
	#define Subtract(c,a,b) c = a - b
#endif

// Multiply two complex variables (used in the defines below)
#if PRECISION == 3232 || PRECISION == 6464
	#define MulReal(a,b) a[0]*b[0] - a[1]*b[1]
	#define MulImag(a,b) a[0]*b[1] + a[1]*b[0]
#endif

// The scalar multiply function
#if PRECISION == 3232 || PRECISION == 6464
	#define Multiply(c,a,b) c = real(MulReal(a,b), MulImag(a,b))
#else
	#define Multiply(c,a,b) c = a * b
#endif

#define vMultiply(c, a, b, vWidth) \
{ \
	UNROLL(vWidth) \
	for (uint i = 0; i < vWidth; i += 1) \
	{ \
		Multiply(c.s[i], a.s[i], b.s[i]); \
	} \
}

// c is vector, a is scalar, b is vector
#define vsMultiply(c, a, b, vWidth) \
{ \
	UNROLL(vWidth) \
	for (uint i = 0; i < vWidth; i += 1) \
	{ \
		Multiply(c.s[i], a, b.s[i]); \
	} \
}

// c is vector, a is scalar, b is vector
#define vsMultiplyAdd(c, a, b, vWidth) \
{ \
	UNROLL(vWidth) \
	for (uint i = 0; i < vWidth; i += 1) \
	{ \
		MultiplyAdd(c.s[i], a, b.s[i]); \
	} \
}

#define vSetToZero(v, vWidth) \
{ \
	UNROLL(vWidth) \
	for (uint i = 0; i < vWidth; i += 1) \
	{ \
		SetToZero(v.s[i]); \
	} \
}

// The scalar multiply-add function
#if PRECISION == 3232 || PRECISION == 6464
	#define MultiplyAdd(c,a,b) c += real(MulReal(a,b), MulImag(a,b))
#else
	#if 0 //USE_CL_MAD == 1
		#define MultiplyAdd(c,a,b) c = mad(a, b, c)
	#else
		#define MultiplyAdd(c,a,b) c += (a * b)
	#endif
#endif

// The scalar multiply-subtract function
#if PRECISION == 3232 || PRECISION == 6464
	#define MultiplySubtract(c,a,b) c -= real(MulReal(a,b), MulImag(a,b))
#else
	#define MultiplySubtract(c,a,b) c -= (a * b)
#endif

// The scalar division function: full division
#if PRECISION == 3232 || PRECISION == 6464
	#define DivideFull(c,a,b) singlereal num_x = (a.x * b.x) + (a.y * b.y); singlereal num_y = (a.y * b.x) - (a.x * b.y); singlereal denom = (b.x * b.x) + (b.y * b.y); c = real(num_x / denom, num_y / denom)
#else
	#define DivideFull(c,a,b) c = a / b
#endif

// The scalar AXPBY function
#if PRECISION == 3232 || PRECISION == 6464
	//#define AXPBY(e,a,b,c,d) e.x = MulReal(a,b) + MulReal(c,d); e.y = MulImag(a,b) + MulImag(c,d)
	#define AXPBY(e,a,b,c,d) e = real(MulReal(a,b) + MulReal(c,d), MulImag(a,b) + MulImag(c,d))
#else
	#define AXPBY(e,a,b,c,d) e = a*b + c*d
#endif

// The complex conjugate operation for complex transforms
#if PRECISION == 3232 || PRECISION == 6464
	#define COMPLEX_CONJUGATE(value) value.x = value.x; value.y = -value.y
#else
	#define COMPLEX_CONJUGATE(value) 
#endif

// =================================================================================================

// Macro for storing and loading, to accomodate BDA
#if USE_BDA
	// this needs to be changed, but I forget how it works
	#define INDEX(buf, idx) buf[idx]
#else
	#define INDEX(buf, idx) buf[idx]
#endif

// =================================================================================================

// vector load methods
#define vload2_single_alignment(index, buf) real2(INDEX(buf, index), INDEX(buf, index+1))
#define vload4_signle_alignment(index, buf) real4(INDEX(buf, index), INDEX(buf, index+1), INDEX(buf, index+2), INDEX(buf, index+3))

#define vload2(index, buf) real2(real[2](INDEX(buf, index), INDEX(buf, index+1)))
#define vload4(index, buf) real4(real[4](INDEX(buf, index), INDEX(buf, index+1), INDEX(buf, index+2), INDEX(buf, index+3)))
#define vload8(index, buf) real8(real[8](INDEX(buf, index), INDEX(buf, index+1), INDEX(buf, index+2),\
	INDEX(buf, index+3), INDEX(buf, index+4), INDEX(buf, index+5), INDEX(buf, index+6), INDEX(buf, index+7)))
#define vload16(index, buf) real16(real[16](INDEX(buf, index), INDEX(buf, index+1), INDEX(buf, index+2),\
	INDEX(buf, index+3), INDEX(buf, index+4), INDEX(buf, index+5), INDEX(buf, index+6), INDEX(buf, index+7),\
	INDEX(buf, index+8), INDEX(buf, index+9), INDEX(buf, index+10), INDEX(buf, index+11), INDEX(buf, index+12),\
	INDEX(buf, index+13), INDEX(buf, index+14), INDEX(buf, index+15) ))
	
#define vloadN(index, buf, N) vload##N(index, buf)

#define vTranspose(dst, src, vWidth) \
{ \
	UNROLL(vWidth) \
	for (uint i = 0; i < vWidth; i += 1) \
	{ \
		UNROLL(vWidth) \
		for (uint j = 0; j < vWidth; j += 1) dst[i].s[j] = src[j].s[i]; \
	} \
}

// =================================================================================================

// Shuffled workgroup indices to avoid partition camping, see below. For specific devices, this is
// enabled (see src/routine.cc).
#ifndef USE_STAGGERED_INDICES
	#define USE_STAGGERED_INDICES 0
#endif

// because I am extremely lazy hehe
#define get_global_id(dim) int(gl_GlobalInvocationID[dim])
#define get_local_id(dim) int(gl_LocalInvocationID[dim])
#define get_group_id(dim) int(gl_WorkGroupID[dim])
#define get_global_size(idx) int(gl_NumWorkGroups[idx] * gl_WorkGroupSize[idx])
#define get_local_size(idx) int(gl_WorkGroupSize[idx])
#define get_num_groups(dim) int(gl_NumWorkGroups[dim])

// Staggered/shuffled group indices to avoid partition camping (AMD GPUs). Formula's are taken from:
// http://docs.nvidia.com/cuda/samples/6_Advanced/transpose/doc/MatrixTranspose.pdf
// More details: https://github.com/CNugteren/CLBlast/issues/53
#if USE_STAGGERED_INDICES == 1 && GEMMK == 0
	int GetGroupIDFlat() {
		return get_group_id(0) + get_num_groups(0) * get_group_id(1);
		//return gl_WorkGroupID.x + gl_NumWorkGroups.x * gl_WorkGroupID.y;
	}
	int GetGroupID1() {
		return (GetGroupIDFlat()) % get_num_groups(1);
		//return int((GetGroupIDFlat()) % gl_NumWorkGroups.y);
	}
	int GetGroupID0() {
		return ((GetGroupIDFlat() / get_num_groups(1)) + GetGroupID1()) % get_num_groups(0);
		//return int(((GetGroupIDFlat() / gl_NumWorkGroups.y) + GetGroupID1()) % gl_WorkGroupSize.x);
	}
#else
	int GetGroupID1() { return get_group_id(1); }
	//int GetGroupID1() { return int(gl_WorkGroupID.y); }
	int GetGroupID0() { return get_group_id(0); }
	//int GetGroupID0() { return int(gl_WorkGroupID.x); }
#endif

// =================================================================================================

// End of the C++11 raw string literal
#endif

// =================================================================================================


// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the common functions and parameters specific for level 3 BLAS kernels.
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
#ifndef LEVEL3_GLSL
#define LEVEL3_GLSL
// =================================================================================================

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library.

// For the 'fast' copy kernel
#ifndef COPY_DIMX
	#define COPY_DIMX 8			// Local workgroup size in the first dimension (x)
#endif
#ifndef COPY_DIMY
	#define COPY_DIMY 8			// Local workgroup size in the second dimension (y)
#endif
#ifndef COPY_WPT
	#define COPY_WPT 1			 // Work per thread in the first dimension (x)
#endif
#ifndef COPY_VW
	#define COPY_VW 1				// Vector width in the second dimension (y)
#endif

// For the padding/copy kernels and the conversion kernels
#ifndef PAD_DIMX
	#define PAD_DIMX 8			// Local workgroup size in the first dimension (x)
#endif
#ifndef PAD_DIMY
	#define PAD_DIMY 8			// Local workgroup size in the second dimension (y)
#endif
#ifndef PAD_WPTX
	#define PAD_WPTX 1			// Work per thread in the first dimension (x)
#endif
#ifndef PAD_WPTY
	#define PAD_WPTY 1			// Work per thread in the second dimension (y)
#endif

// For the 'fast' transpose kernel
#ifndef TRA_DIM
	#define TRA_DIM 8			 // Number of local threads in the two dimensions (x,y)
#endif
#ifndef TRA_WPT
	#define TRA_WPT 1			 // Work per thread in one dimension and vector-width in the other
#endif
#ifndef TRA_PAD
	#define TRA_PAD 0			 // Padding of the local memory to avoid bank-conflicts
#endif
#ifndef TRA_SHUFFLE
	#define TRA_SHUFFLE 0	 // Shuffling of the global indices to avoid global memory bank-conflicts
#endif

// For the padding/transpose kernels
#ifndef PADTRA_TILE
	#define PADTRA_TILE 8	 // Number of local threads in the two dimensions (x,y)
#endif
#ifndef PADTRA_WPT
	#define PADTRA_WPT 1		// Amount of work per thread
#endif
#ifndef PADTRA_PAD
	#define PADTRA_PAD 0		// Padding of the local memory to avoid bank-conflicts
#endif

// =================================================================================================
#endif
// End of the C++11 raw string literal

// =================================================================================================

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library. Note that all parameters here have a
// suffix 'D' to denote that they are for the 'direct' version of the GEMM kernel.
#ifndef WGD
	#define WGD 8			// Tile-size in dimension M, N, and K (e.g. 8, 16, 32, 64)
#endif
#ifndef MDIMCD
	#define MDIMCD 8		// Threads per workgroup in M-dimension (e.g. 8, 16, 32)
#endif
#ifndef NDIMCD
	#define NDIMCD 8		// Threads per workgroup in N-dimension (e.g. 8, 16, 32)
#endif
#ifndef MDIMAD
	#define MDIMAD 8		// Re-shaped tile dimension of matrix A: KDIMAD * MDIMAD
#endif
#ifndef NDIMBD
	#define NDIMBD 8		// Re-shaped tile dimension of matrix B: KDIMBD * NDIMBD
#endif
#ifndef KWID
	#define KWID 1			// Unroll factor of the WGD loop (smaller or equal than WGD)
#endif
#ifndef VWMD
	#define VWMD 1			// Vector width of matrices A and C
#endif
#ifndef VWND
	#define VWND 1			// Vector width of matrix B
#endif
#ifndef PADA
	#define PADA 1			// Local memory padding for matrix A
#endif
#ifndef PADB
	#define PADB 1			// Local memory padding for matrix B
#endif

// Helper parameters based on the above tuning parameters
#define MWID (WGD/MDIMCD)								// Work per work-item (M-dimension)
#define NWID (WGD/NDIMCD)								// Work per work-item (N-dimension)
#define KDIMAD ((MDIMCD*NDIMCD)/(MDIMAD)) // Re-shaped tile dimension of matrix A: KDIMAD * MDIMAD
#define KDIMBD ((MDIMCD*NDIMCD)/(NDIMBD)) // Re-shaped tile dimension of matrix B: KDIMBD * NDIMBD
#define MWAD (WGD/MDIMAD)								// Amount of loads-per-thread for matrix A (M-dimension)
#define KWAD (WGD/KDIMAD)								// Amount of loads-per-thread for matrix A (K-dimension)
#define KWBD (WGD/KDIMBD)								// Amount of loads-per-thread for matrix B (K-dimension)
#define NWBD (WGD/NDIMBD)								// Amount of loads-per-thread for matrix B (N-dimension)

// =================================================================================================

// Data-widths in dimension M
#if VWMD == 1
		#define realMD real
#elif VWMD == 2
		#define realMD real2
#elif VWMD == 4
		#define realMD real4
#elif VWMD == 8
		#define realMD real8
#elif VWMD == 16
		#define realMD real16
#endif

// Data-widths in dimension N
#if VWND == 1
		#define realND real
#elif VWND == 2
		#define realND real2
#elif VWND == 4
		#define realND real4
#elif VWND == 8
		#define realND real8
#elif VWND == 16
		#define realND real16
#endif

// =================================================================================================

// Loads global off-chip memory into thread-private register files. This function is specific for
// loading the A input matrix.
#define GlobalToPrivateDirectA(result, agms, _mi, a_ld, a_offset, idm, idk, a_transpose, a_conjugate) \
{ \
	const int a_index = bool(a_transpose) ? (idm + _mi)*a_ld + idk : idk*a_ld + (idm + _mi); \
	result = agms[a_index + a_offset]; \
	if (bool(a_conjugate)) { COMPLEX_CONJUGATE(result); } \
} \

// Same as above, but now for the B input matrix
#define GlobalToPrivateDirectB(result, bgms, _ni, b_ld, b_offset, idn, idk, b_transpose, b_conjugate) \
{ \
	const int b_index = bool(b_transpose) ? (idn + _ni)*b_ld + idk : idk*b_ld + (idn + _ni); \
	result = bgms[b_index + b_offset]; \
	if (bool(b_conjugate)) { COMPLEX_CONJUGATE(result); } \
}

// Loads global off-chip memory into thread-private register files. This function is specific for
// loading the A input matrix. This is the same as above but now includes a bounds check.
#define GlobalToPrivateCheckedA(result, agms, _mi, a_ld, a_offset, idm, idk, a_transpose, a_conjugate, kSizeM) \
{ \
	if (idm + _mi < kSizeM) { \
		const int a_index = bool(a_transpose) ? (idm + _mi)*a_ld + idk : idk*a_ld + (idm + _mi); \
		result = agms[a_index + a_offset]; \
		if (bool(a_conjugate)) { COMPLEX_CONJUGATE(result); } \
	} \
	else { SetToZero(result); } \
}

// Same as above, but now for the B input matrix
#define GlobalToPrivateCheckedB(result, bgms, _ni, b_ld, b_offset, idn, idk, b_transpose, b_conjugate, kSizeN) \
{ \
	if (idn + _ni < kSizeN) { \
		const int b_index = bool(b_transpose) ? (idn + _ni)*b_ld + idk : idk*b_ld + (idn + _ni); \
		result = bgms[b_index + b_offset]; \
		if (bool(b_conjugate)) { COMPLEX_CONJUGATE(result); } \
	} \
	else { SetToZero(result); } \
}

// =================================================================================================

// Caches on-chip local memory into per-thread private memory (registers). This function is specific
// for caching the A input matrix.
// (has to be a macro because GLSL doesn't like passing local memory as arguments
#define LocalToPrivateDirectA(result, alm, _mi, kg, a_transpose) \
{ \
	const int mg = _mi + get_local_id(0)*MWID; \
	const int index = bool(a_transpose) ? mg*(WGD + PADA) + kg : kg*(WGD + PADA) + mg; \
	result = alm[index]; \
} \

// Same as above, but now for the B input matrix
#define LocalToPrivateDirectB(result, blm, _ni, kg, b_transpose) \
{ \
	const int ng = _ni + get_local_id(1)*NWID; \
	const int index = bool(b_transpose) ? ng*(WGD + PADB) + kg : kg*(WGD + PADB) + ng; \
	result = blm[index]; \
} \

// =================================================================================================

// Merges the results in Cpm with the global array in Cgm. This also performs the multiplication
// with the constants: Cgm = alpha*A*B + beta*Cgm = alpha*Cpm + beta*Cgm
#define StoreResultsDirect(cgm, c_value, _mi, _ni, idm, idn, alpha, beta, c_ld, c_offset, c_transpose) \
{ \
	int c_index = bool(c_transpose) ? (idm + _mi)*c_ld + (idn + _ni) : (idn + _ni)*c_ld + (idm + _mi); \
	real result; \
	if (IsZero(beta)) \
	{ \
		Multiply(result, alpha, c_value); \
	} \
	else \
	{ \
		AXPBY(result, alpha, c_value, beta, cgm[c_index + c_offset]); \
	} \
	cgm[c_index + c_offset] = result; \
} \

// Merges the results in Cpm with the global array in Cgm. This also performs the multiplication
// with the constants: Cgm = alpha*A*B + beta*Cgm = alpha*Cpm + beta*Cgm
#define StoreResultsChecked(cgm, c_value, _mi, _ni, idm, idn, kSizeM, kSizeN, alpha, beta, c_ld, c_offset, c_transpose) \
{ \
	if ((idm + _mi) < kSizeM && (idn + _ni) < kSizeN) { \
		int c_index = bool(c_transpose) ? (idm + _mi)*c_ld + (idn + _ni) : (idn + _ni)*c_ld + (idm + _mi); \
		real result; \
		if (IsZero(beta)) { \
			Multiply(result, alpha, c_value); \
		} \
		else { \
			AXPBY(result, alpha, c_value, beta, cgm[c_index + c_offset]); \
		} \
		cgm[c_index + c_offset] = result; \
	} \
} \

// =================================================================================================

// End of the C++11 raw string literal

// =================================================================================================

// =================================================================================================

// because preprocessor conditions and function-like macros don't like each other...
ivec2 getIndexForGlobalToLocalM()
{
	#if MDIMCD == MDIMAD
		const int la0 = get_local_id(0);
		const int la1 = get_local_id(1);
	#else
		const int tid = get_local_id(0) + MDIMCD*get_local_id(1);
		const int la0 = tid % MDIMAD;
		const int la1 = tid / MDIMAD;
	#endif
	return ivec2(la0, la1);
}

ivec2 getIndexForGlobalToLocalN()
{
	#if MDIMCD == NDIMBD
		const int lb0 = get_local_id(0);
		const int lb1 = get_local_id(1);
	#else
		const int tid = get_local_id(0) + MDIMCD*get_local_id(1);
		const int lb0 = tid % NDIMBD;
		const int lb1 = tid / NDIMBD;
	#endif
	return ivec2(lb0, lb1);
}

// looks more and more like we are going to have to define multiple macros for each vector size :c
// wait a second...I think I know what to do!
#if VWMD == 1
	#define StoreMDInLocal(lm, lm_idx, value) lm[lm_idx] = value
#else
	#define StoreMDInLocal(lm, lm_idx, value) \
	{ \
		UNROLL(VWMD) \
		for (uint iv = 0; iv < VWMD; iv += 1) \
		{ \
			lm[lm_idx + iv] = value.s[iv]; \
		} \
	}
#endif

// same as above but for ND
#if VWND == 1
	#define StoreNDInLocal(lm, lm_idx, value) lm[lm_idx] = value
#else
	#define StoreNDInLocal(lm, lm_idx, value) \
	{ \
		UNROLL(VWND) \
		for (uint iv = 0; iv < VWND; iv += 1) \
		{ \
			lm[lm_idx + iv] = value.s[iv]; \
		} \
	}
#endif

// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for
// caching the A input matrix.
#define GlobalToLocalDirectA(agm, alm, a_ld, a_offset, kwg, a_transpose, a_conjugate) \
{ \
	const ivec2 la = getIndexForGlobalToLocalM(); \
	const int la0 = la.x; \
	const int la1 = la.y; \
	for (int _mia = 0; _mia < MWAD/VWMD; _mia += 1) { \
		for (int _kia = 0; _kia < KWAD; _kia += 1) { \
			int mg = _mia + la0*(MWAD/VWMD); \
			int kg = _kia + la1*KWAD; \
			int idm = bool(a_transpose) ? mg + kwg/VWMD : mg + GetGroupID0()*(WGD/VWMD); \
			int idk = bool(a_transpose) ? kg + GetGroupID0()*WGD : kg + kwg; \
			const realMD avec = agm[idk*(a_ld/VWMD) + idm + (a_offset/VWMD)]; \
			StoreMDInLocal(alm, kg*(WGD + PADA) + mg*VWMD, avec); \
			if (bool(a_conjugate)) { \
				for (int vm=0; vm<VWMD; ++vm) { \
					COMPLEX_CONJUGATE(alm[kg*(WGD + PADA) + mg*VWMD + vm]); \
				} \
			} \
		} \
	} \
} 

// Same as above, but now for the B input matrix
#define GlobalToLocalDirectB(bgm, blm, b_ld, b_offset, kwg, b_transpose, b_conjugate) \
{ \
	const ivec2 lb = getIndexForGlobalToLocalN(); \
	const int lb0 = lb.x; \
	const int lb1 = lb.y; \
	for (int _kib = 0; _kib < KWBD; _kib += 1) { \
		for (int _nib = 0; _nib < NWBD/VWND; _nib += 1) { \
			int ng = _nib + lb0*(NWBD/VWND); \
			int kg = _kib + lb1*KWBD; \
			int idn = bool(b_transpose) ? ng + kwg/VWND : ng + GetGroupID1()*(WGD/VWND); \
			int idk = bool(b_transpose) ? kg + GetGroupID1()*WGD : kg + kwg; \
			const realND bvec = bgm[idk*(b_ld/VWND) + idn + (b_offset/VWND)]; \
			StoreNDInLocal(blm, kg*(WGD + PADB) + ng*VWND, bvec); \
			if (bool(b_conjugate)) { \
				for (int _vn = 0; _vn < VWND; _vn += 1) { \
					COMPLEX_CONJUGATE(blm[kg*(WGD + PADB) + ng*VWND + _vn]); \
				} \
			} \
		} \
	} \
}

// =================================================================================================

// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for
// caching the A input matrix. In contrast to the functions above, this function performs doesn't
// use the vector data-types.
#define GlobalToLocalScalarA(agms, alm, a_ld, a_offset, kwg, a_transpose, a_conjugate) \
{ \
	const ivec2 la = getIndexForGlobalToLocalM(); \
	const int la0 = la.x; \
	const int la1 = la.y; \
	for (int _mia = 0; _mia < MWAD; _mia += 1) { \
		for (int _kia = 0; _kia < KWAD; _kia += 1) { \
			int mg = _mia + la0*MWAD; \
			int kg = _kia + la1*KWAD; \
			int idm = bool(a_transpose) ? mg + kwg : mg + GetGroupID0()*WGD; \
			int idk = bool(a_transpose) ? kg + GetGroupID0()*WGD : kg + kwg; \
			real result = agms[idk*a_ld + idm + a_offset]; \
			if (bool(a_conjugate)) { COMPLEX_CONJUGATE(result); } \
			alm[kg*(WGD + PADA) + mg] = result; \
		} \
	} \
}

// Same as above, but now for the B input matrix
#define GlobalToLocalScalarB(bgms, blm, b_ld, b_offset, kwg, b_transpose, b_conjugate) \
{ \
	const ivec2 lb = getIndexForGlobalToLocalN(); \
	const int lb0 = lb.x; \
	const int lb1 = lb.y; \
	for (int _kib = 0; _kib < KWBD; _kib += 1) { \
		for (int _nib = 0; _nib < NWBD; _nib += 1) { \
			int ng = _nib + lb0*NWBD; \
			int kg = _kib + lb1*KWBD; \
			int idn = bool(b_transpose) ? ng + kwg : ng + GetGroupID1()*WGD; \
			int idk = bool(b_transpose) ? kg + GetGroupID1()*WGD : kg + kwg; \
			real result = bgms[idk*b_ld + idn + b_offset]; \
			if (bool(b_conjugate)) { COMPLEX_CONJUGATE(result); } \
			blm[kg*(WGD + PADB) + ng] = result; \
		} \
	} \
}

// =================================================================================================

// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for
// caching the A input matrix. In contrast to the functions above, this function performs bounds
// checks and doesn't use the vector data-types.
#define GlobalToLocalCheckedA(agms, alm, a_ld, a_offset, kwg, a_transpose, a_conjugate, kSizeM, kSizeK) \
{ \
	const ivec2 la = getIndexForGlobalToLocalM(); \
	const int la0 = la.x; \
	const int la1 = la.y; \
	for (int _mia = 0; _mia < MWAD; _mia += 1) { \
		for (int _kia = 0; _kia < KWAD; _kia += 1) { \
			int mg = _mia + la0*MWAD; \
			int kg = _kia + la1*KWAD; \
			int idm = bool(a_transpose) ? mg + kwg : mg + GetGroupID0()*WGD; \
			int idk = bool(a_transpose) ? kg + GetGroupID0()*WGD : kg + kwg; \
			const bool condition = bool(a_transpose) ? (idm < kSizeK) && (idk < kSizeM) : (idm < kSizeM) && (idk < kSizeK); \
			if (condition) { \
				real result = agms[idk*a_ld + idm + a_offset]; \
				if (bool(a_conjugate) ) { COMPLEX_CONJUGATE(result); } \
				alm[kg*(WGD + PADA) + mg] = result; \
			} \
			else { \
				SetToZero(alm[kg*(WGD + PADA) + mg]); \
			} \
		} \
	} \
}

// Same as above, but now for the B input matrix
#define GlobalToLocalCheckedB(bgms, blm, b_ld, b_offset, kwg, b_transpose, b_conjugate, kSizeN, kSizeK) \
{ \
	const ivec2 lb = getIndexForGlobalToLocalN(); \
	const int lb0 = lb.x; \
	const int lb1 = lb.y; \
	for (int _kib = 0; _kib < KWBD; _kib += 1) { \
		for (int _nib = 0; _nib < NWBD; _nib += 1) { \
			int ng = _nib + lb0*NWBD; \
			int kg = _kib + lb1*KWBD; \
			int idn = bool(b_transpose) ? ng + kwg : ng + GetGroupID1()*WGD; \
			int idk = bool(b_transpose) ? kg + GetGroupID1()*WGD : kg + kwg; \
			const bool condition = bool(b_transpose) ? (idn < kSizeK) && (idk < kSizeN) : (idn < kSizeN) && (idk < kSizeK); \
			if (condition) { \
				real result = bgms[idk*b_ld + idn + b_offset]; \
				if (bool(b_conjugate)) { COMPLEX_CONJUGATE(result); } \
				blm[kg*(WGD + PADB) + ng] = result; \
			} \
			else { \
				SetToZero(blm[kg*(WGD + PADB) + ng]); \
			} \
		} \
	} \
}

// =================================================================================================

// End of the C++11 raw string literal

// =================================================================================================

// =================================================================================================

// global and shared memory declarations go here, since they are shared across all kernels c:
// *gm and *gms both bind to the same underlying memory
layout(binding = 0) buffer agm_buf { realMD agm[]; };
layout(binding = 1) buffer bgm_buf { realND bgm[]; };
layout(binding = 2) buffer cgm_buf { real cgm[]; };

layout(binding = 3) buffer agms_buf { real agms[]; };
layout(binding = 4) buffer bgms_buf { real bgms[]; };

shared real alm[WGD * (WGD + PADA)];
shared real blm[WGD * (WGD + PADB)];

#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = MDIMCD, local_size_y = NDIMCD, local_size_z = 1) in;
#endif

// Main body of the kernel. This is the direct version without pre/post processing and restrictions.
void XgemmDirect(const int kSizeM, const int kSizeN, const int kSizeK, const real_arg arg_alpha,
		const real_arg arg_beta,
#if USE_BDA
		const __global realMD* restrict agm,
#endif
		const int a_offset, const int a_ld,
#if USE_BDA
		const __global realND* restrict bgm,
#endif
		const int b_offset, const int b_ld,
#if USE_BDA
		__global real* cgm,
#endif
		const int c_offset, const int c_ld,
		// no local memory args allowed :c
		//LOCAL_PTR real* alm, LOCAL_PTR real* blm,
		const int a_transpose, const int b_transpose, const int c_transpose,
		const int a_conjugate, const int b_conjugate)
{
	const real alpha = GetRealArg(arg_alpha);
	const real beta = GetRealArg(arg_beta);

	// Extra pointers to scalar versions of global memory
#if USE_BDA
	const __global real* restrict agms = (const __global real* restrict) agm;
	const __global real* restrict bgms = (const __global real* restrict) bgm;
#endif

	// Allocates workitem-private memory (registers)
	
	real apd[MWID];
	
	real bpd[NWID];
	
	real cpd[NWID * MWID];

	// Initializes the accumulation registers
	
	for (int _mi = 0; _mi < MWID; _mi += 1) {
		
		for (int _ni = 0; _ni < NWID; _ni += 1) {
			SetToZero(cpd[_ni * MWID + _mi]);
		}
	}

	// The faster version of GEMM is not allowed on the (incomplete) borders. Therefore, this section
	// processes only the main parts: output blocks of WGD by WGD.
	const int idm = get_local_id(0) * MWID + GetGroupID0() * WGD;
	const int idn = get_local_id(1) * NWID + GetGroupID1() * WGD;
	if ((idm < (kSizeM/WGD)*WGD) && (idn < (kSizeN/WGD)*WGD)) {

		// Loops over all complete workgroup tiles (K-dimension)
		int kwg = 0;
		for (; kwg < (kSizeK/WGD) * WGD; kwg += WGD) {

			// Loads data: off-chip --> local (matrix A and B)
			if (a_ld % VWMD == 0 && a_offset % VWMD == 0) {
				GlobalToLocalDirectA(agm, alm, a_ld, a_offset, kwg, a_transpose, a_conjugate);
			}
			else {
				GlobalToLocalScalarA(agms, alm, a_ld, a_offset, kwg, a_transpose, a_conjugate);
			}
			if (b_ld % VWND == 0 && b_offset % VWND == 0) {
				GlobalToLocalDirectB(bgm, blm, b_ld, b_offset, kwg, b_transpose, b_conjugate);
			}
			else {
				GlobalToLocalScalarB(bgms, blm, b_ld, b_offset, kwg, b_transpose, b_conjugate);
			}
			barrier();

			// Loops over all workitem tiles, unrolled by a factor KWID
			for (int pwi = 0; pwi < WGD; pwi += KWID) {
				
				for (int _pit = 0; _pit < KWID; _pit += 1) {
					int kg = pwi + _pit;

					// Loads data: local --> private (matrix A and B)
					
					for (int _mi = 0; _mi < MWID; _mi += 1) {
						 LocalToPrivateDirectA(apd[_mi], alm, _mi, kg, a_transpose);
					}
					
					for (int _ni = 0; _ni < NWID; _ni += 1) {
						LocalToPrivateDirectB(bpd[_ni], blm, _ni, kg, b_transpose);
					}

					// Performs the accumulation (Cpmd += Apmd * Bpmd)
					
					for (int _ni = 0; _ni < NWID; _ni += 1) {
						
						for (int _mi = 0; _mi < MWID; _mi += 1) {
							MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);
						}
					}
				}
			}
			barrier();
		}

		// Loop over the remaining part (incomplete tile in K-dimension)
		for (; kwg < kSizeK; ++kwg) {

			// Loads data: off-chip --> private (matrix A and B)
			
			for (int _mi = 0; _mi < MWID; _mi += 1) {
				GlobalToPrivateDirectA(apd[_mi], agms, _mi, a_ld, a_offset, idm, kwg, a_transpose, a_conjugate);
			}
			
			for (int _ni = 0; _ni < NWID; _ni += 1) {
				GlobalToPrivateDirectB(bpd[_ni], bgms, _ni, b_ld, b_offset, idn, kwg, b_transpose, b_conjugate);
			}

			// Performs the accumulation (Cpmd += Apmd * Bpmd)
			
			for (int _ni = 0; _ni < NWID; _ni += 1) {
				
				for (int _mi = 0; _mi < MWID; _mi += 1) {
					MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);
				}
			}
		}

		// Stores a tile of results and performs the multiplication with alpha and beta
		
		for (int _ni = 0; _ni < NWID; _ni += 1) {
			
			for (int _mi = 0; _mi < MWID; _mi += 1) {
				StoreResultsDirect(cgm, cpd[_ni * MWID + _mi], _mi, _ni, idm, idn,
													 alpha, beta, c_ld, c_offset, c_transpose);
			}
		}
	}

	// Simple but slower version for the parts on the edge (incomplete tiles in M and N-dimensions)
	else {

		// Loops over all complete workgroup tiles (K-dimension)
		int kwg = 0;
		for (; kwg < (kSizeK/WGD) * WGD; kwg+=WGD) {

			// Loads data: off-chip --> local (matrix A and B)
			GlobalToLocalCheckedA(agms, alm, a_ld, a_offset, kwg, a_transpose, a_conjugate, kSizeM, kSizeK);
			GlobalToLocalCheckedB(bgms, blm, b_ld, b_offset, kwg, b_transpose, b_conjugate, kSizeN, kSizeK);
			barrier();

			// Loops over all workitem tiles, unrolled by a factor KWID
			for (int pwi = 0; pwi < WGD; pwi += KWID) {
				
				for (int _pit = 0; _pit < KWID; _pit += 1) {
					int kg = pwi + _pit;

					// Loads data: local --> private (matrix A and B)
					
					for (int _mi = 0; _mi < MWID; _mi += 1) {
						LocalToPrivateDirectA(apd[_mi], alm, _mi, kg, a_transpose);
					}
					
					for (int _ni = 0; _ni < NWID; _ni += 1) {
						LocalToPrivateDirectB(bpd[_ni], blm, _ni, kg, b_transpose);
					}

					// Performs the accumulation (C += A * B)
					
					for (int _ni = 0; _ni < NWID; _ni += 1) {
						
						for (int _mi = 0; _mi < MWID; _mi += 1) {
							MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);
						}
					}
				}
			}
			barrier();
		}

		// Loop over the remaining part (incomplete tile in K-dimension)
		for (; kwg < kSizeK; ++kwg) {

			// Loads data: off-chip --> private (matrix A and B)
			
			for (int _mi = 0; _mi < MWID; _mi += 1) {
				GlobalToPrivateCheckedA(apd[_mi], agms, _mi, a_ld, a_offset, idm, kwg, a_transpose, a_conjugate, kSizeM);
			}
			
			for (int _ni = 0; _ni < NWID; _ni += 1) {
				GlobalToPrivateCheckedB(bpd[_ni], bgms, _ni, b_ld, b_offset, idn, kwg, b_transpose, b_conjugate, kSizeN);
			}

			// Performs the accumulation (C += A * B)
			
			for (int _ni = 0; _ni < NWID; _ni += 1) {
				
				for (int _mi = 0; _mi < MWID; _mi += 1) {
					MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);
				}
			}
		}

		// Stores a tile of results and performs the multiplication with alpha and beta
		
		for (int _ni = 0; _ni < NWID; _ni += 1) {
			
			for (int _mi = 0; _mi < MWID; _mi += 1) {
				StoreResultsChecked(cgm, cpd[_ni * MWID + _mi], _mi, _ni, idm, idn, kSizeM, kSizeN,
														alpha, beta, c_ld, c_offset, c_transpose);
			}
		}
	}
}

// =================================================================================================

// End of the C++11 raw string literal

// =================================================================================================

// =================================================================================================

// =================================================================================================

// Direct version of the strided-batched GEMM kernel with [A, B] = [non-transposed, non-transposed]
#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = MDIMCD, local_size_y = NDIMCD, local_size_z = 1) in;
#endif
layout(push_constant) uniform XgemmDirectStridedBatchedNN
{
	int kSizeM; int kSizeN; int kSizeK;
	real_arg arg_alpha; real_arg arg_beta;
#if USE_BDA
	__global realMD* restrict agm;
#endif
	int a_offset; int a_ld; int a_stride;
#if USE_BDA
	__global realND* restrict bgm;
#endif
	int b_offset; int b_ld; int b_stride;
#if USE_BDA
	__global real* cgm;
#endif
	int c_offset; int c_ld; int c_stride;
	int c_transpose; int a_conjugate; int b_conjugate;
} args;

void main()																 
{
	const int batch = get_group_id(2);
	const int a_offset_batch = args.a_offset + args.a_stride * batch;
	const int b_offset_batch = args.b_offset + args.b_stride * batch;
	const int c_offset_batch = args.c_offset + args.c_stride * batch;

	XgemmDirect(args.kSizeM, args.kSizeN, args.kSizeK, args.arg_alpha, args.arg_beta,
#if USE_BDA
		agm,
#endif
		a_offset_batch, args.a_ld,
#if USE_BDA
		bgm,
#endif
		b_offset_batch, args.b_ld,
#if USE_BDA
		cgm,
#endif
		c_offset_batch, args.c_ld,
		//alm, blm,
		0, 0, args.c_transpose, args.a_conjugate, args.b_conjugate);
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
