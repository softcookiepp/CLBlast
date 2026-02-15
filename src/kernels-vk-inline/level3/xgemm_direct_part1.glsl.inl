
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
R"(
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
)"

// =================================================================================================
