
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 3 of 3 of the GEMM kernel. See part 1 for more information.
//
// =================================================================================================

// literal). Comment-out this line for syntax-highlighting when developing.
R"(
// =================================================================================================

// global and shared memory declarations go here, since they are shared across all kernels c:
// *gm and *gms both bind to the same underlying memory
layout(binding = 0, std430) buffer arg_alphas_buf { real_arg arg_alphas[]; };
layout(binding = 1, std430) buffer arg_betas_buf { real_arg arg_betas[]; };
layout(binding = 2) buffer agm_buf { realMD agm[]; };
layout(binding = 3) buffer a_offsets_buf { int a_offsets[]; };
layout(binding = 4) buffer bgm_buf { realND bgm[]; };
layout(binding = 5) buffer b_offsets_buf { int b_offsets[]; };
layout(binding = 6) buffer cgm_buf { real cgm[]; };
layout(binding = 7) buffer c_offsets_buf { int c_offsets[]; };

layout(binding = 8) buffer agms_buf { real agms[]; };
layout(binding = 9) buffer bgms_buf { real bgms[]; };

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
)"

// =================================================================================================
