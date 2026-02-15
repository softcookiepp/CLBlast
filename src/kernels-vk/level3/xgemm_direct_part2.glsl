
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 2 of 3 of the GEMM kernel. See part 1 for more information.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
//R"(
#ifndef XGEMM_DIRECT_PART2_GLSL
#define XGEMM_DIRECT_PART2_GLSL
#include "xgemm_direct_part1.glsl"
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
#endif
// End of the C++11 raw string literal
//)"

// =================================================================================================
