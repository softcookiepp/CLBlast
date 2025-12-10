
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 2 of 3 of the GEMM kernel. See part 1 for more information.
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
R"(

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
#elif VWMD == 2 || VWMD == 4
	#if PRECISION == 3232 || PRECISION == 6464
		#if VWMD == 2
			#define StoreMDInLocal(lm, lm_idx, value)
			{ \
				lm[lm_idx + 0] = value.x; \
				lm[lm_idx + 1] = value.y; \
			}
		#elif VWMD == 4
			#define StoreMDInLocal(lm, lm_idx, value)
			{ \
				lm[lm_idx + 0] = value.x; \
				lm[lm_idx + 1] = value.y; \
				lm[lm_idx + 2] = value.z; \
				lm[lm_idx + 3] = value.w; \
			}
		#endif
	#else
		#define StoreMDInLocal(lm, lm_idx, value)
		{ \
			for (uint i = 0; i < VWMD; i += 1) \
			{ \
				lm[lm_idx + i] = value[i]; \
			} \
		}
	#endif
#else
	// 8 or 16
	#if PRECISION == 3232 || PRECISION == 6464
		#if VWMD == 8
			#define StoreMDInLocal(lm, lm_idx, value)
			{ \
				lm[lm_idx + 0] = value.s0; \
				lm[lm_idx + 1] = value.s1; \
				lm[lm_idx + 2] = value.s2; \
				lm[lm_idx + 3] = value.s3; \
				lm[lm_idx + 4] = value.s4; \
				lm[lm_idx + 5] = value.s5; \
				lm[lm_idx + 6] = value.s6; \
				lm[lm_idx + 7] = value.s7; \
			}
		#elif VWMD == 16
			#define StoreMDInLocal(lm, lm_idx, value)
			{ \
				lm[lm_idx + 0] = value.s0; \
				lm[lm_idx + 1] = value.s1; \
				lm[lm_idx + 2] = value.s2; \
				lm[lm_idx + 3] = value.s3; \
				lm[lm_idx + 4] = value.s4; \
				lm[lm_idx + 5] = value.s5; \
				lm[lm_idx + 6] = value.s6; \
				lm[lm_idx + 7] = value.s7; \
				lm[lm_idx + 8] = value.s8; \
				lm[lm_idx + 9] = value.s9; \
				lm[lm_idx + 10] = value.sA; \
				lm[lm_idx + 11] = value.sB; \
				lm[lm_idx + 12] = value.sC; \
				lm[lm_idx + 13] = value.sD; \
				lm[lm_idx + 14] = value.sE; \
				lm[lm_idx + 15] = value.sF; \
			}
		#endif
	// with non-complex precisions,  matrices are used as a workaround for GLSL's lack of large vector support, so that SIMD can still be used to some degree
	#else
		#define StoreMDInLocal(lm, lm_idx, value)
		{ \
			for (uint i = 0; i < VWMD/4; i += 1) \
			{ \
				for (uint j = 0; j < 4; j += 1) \
				{ \
					lm[lm_idx + 4*i + j] = value[i][j]; \
				} \
			} \
		}
	#endif
#endif

// same as above but for ND
#if VWND == 1
	#define StoreNDInLocal(lm, lm_idx, value) lm[lm_idx] = value
#elif VWND == 2 || VWND == 4
	#if PRECISION == 3232 || PRECISION == 6464
		#if VWND == 2
			#define StoreNDInLocal(lm, lm_idx, value)
			{ \
				lm[lm_idx + 0] = value.x; \
				lm[lm_idx + 1] = value.y; \
			}
		#elif VWND == 4
			#define StoreNDInLocal(lm, lm_idx, value)
			{ \
				lm[lm_idx + 0] = value.x; \
				lm[lm_idx + 1] = value.y; \
				lm[lm_idx + 2] = value.z; \
				lm[lm_idx + 3] = value.w; \
			}
		#endif
	#else
		#define StoreNDInLocal(lm, lm_idx, value)
		{ \
			for (uint i = 0; i < VWND; i += 1) \
			{ \
				lm[lm_idx + i] = value[i]; \
			} \
		}
	#endif
#else
	// 8 or 16
	#if PRECISION == 3232 || PRECISION == 6464
		#if VWND == 8
			#define StoreNDInLocal(lm, lm_idx, value)
			{ \
				lm[lm_idx + 0] = value.s0; \
				lm[lm_idx + 1] = value.s1; \
				lm[lm_idx + 2] = value.s2; \
				lm[lm_idx + 3] = value.s3; \
				lm[lm_idx + 4] = value.s4; \
				lm[lm_idx + 5] = value.s5; \
				lm[lm_idx + 6] = value.s6; \
				lm[lm_idx + 7] = value.s7; \
			}
		#elif VWND == 16
			#define StoreNDInLocal(lm, lm_idx, value)
			{ \
				lm[lm_idx + 0] = value.s0; \
				lm[lm_idx + 1] = value.s1; \
				lm[lm_idx + 2] = value.s2; \
				lm[lm_idx + 3] = value.s3; \
				lm[lm_idx + 4] = value.s4; \
				lm[lm_idx + 5] = value.s5; \
				lm[lm_idx + 6] = value.s6; \
				lm[lm_idx + 7] = value.s7; \
				lm[lm_idx + 8] = value.s8; \
				lm[lm_idx + 9] = value.s9; \
				lm[lm_idx + 10] = value.sA; \
				lm[lm_idx + 11] = value.sB; \
				lm[lm_idx + 12] = value.sC; \
				lm[lm_idx + 13] = value.sD; \
				lm[lm_idx + 14] = value.sE; \
				lm[lm_idx + 15] = value.sF; \
			}
		#endif
	// with non-complex precisions,  matrices are used as a workaround for GLSL's lack of large vector support, so that SIMD can still be used to some degree
	#else
		#define StoreNDInLocal(lm, lm_idx, value)
		{ \
			for (uint i = 0; i < VWND/4; i += 1) \
			{ \
				for (uint j = 0; j < 4; j += 1) \
				{ \
					lm[lm_idx + 4*i + j] = value[i][j]; \
				} \
			} \
		}
	#endif
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
)"

// =================================================================================================
