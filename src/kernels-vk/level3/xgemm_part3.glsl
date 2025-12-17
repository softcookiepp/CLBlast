
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 3 of 4 of the GEMM kernel. See part 1 for more information.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
//R"(

// A common interface for subgroup functions
// genuinely no idea how this maps to GLSL as of now; Vulkan probably has entirely different extensions
// We will just have to disable it host-side until a solution is found...
#if USE_SUBGROUP_SHUFFLING == 1

int clblast_get_sub_group_local_id()
{

	// Intel extension 
	#if SUBGROUP_SHUFFLING_INTEL == 1
	return get_sub_group_local_id();
	
	// Nvidia inline PTX
	#elif SUBGROUP_SHUFFLING_NVIDIA_PRE_VOLTA == 1 || SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA == 1
	int ret;
	asm volatile("mov.u32 %0, %%laneid;" : "=r"(ret) );
	return ret;
	#endif 
}

realN clblast_sub_group_shuffle(realN reg, int src) {

	// Intel extension 
	#if SUBGROUP_SHUFFLING_INTEL == 1
	return intel_sub_group_shuffle(reg, src);
	
	// Nvidia inline PTX
	// Volta and later requires .sync shuffle instructions with an extra mask arg
	#elif SUBGROUP_SHUFFLING_NVIDIA_PRE_VOLTA == 1 || SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA == 1
	realN ret;
		#if SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA == 1
		asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(ret): "f"(reg), "r"(src));
		#else
		asm volatile("shfl.idx.b32 %0, %1, %2, 0x1f;" : "=f"(ret): "f"(reg), "r"(src));
		#endif
	return ret;
	#endif
}
#endif

// Main body of the matrix-multiplication algorithm. It calls various (inlined) functions.
void XgemmBody(const int kSizeM, const int kSizeN, const int kSizeK,
#if USE_BDA
	const __global realM* restrict agm, const __global realN* restrict bgm,
	__global realM* cgm,
#else
	int a_offset, int b_offset, int c_offset,
#endif
	const real alpha, const real beta)
{

	// Allocates workitem-private memory (registers)
	#if GEMMK == 0
		
		realM apm[MWI/VWM]; // MWI * 1
		
		realN bpm[NWI/VWN]; // 1 * NWI
	#elif GEMMK == 1
		#if USE_SUBGROUP_SHUFFLING == 1
			
			realN apm[KREG/VWN]; // KREG (subgroup shuffling in NWI dimension)
		#else
			
			realN apm[NWI*(KREG/VWN)]; // NWI * KREG
		#endif
		
		realM bpm[KREG*(MWI/VWM)]; // KREG * MWI
	#endif
	
	realM cpm[NWI*(MWI/VWM)]; // NWI * MWI

	#if GEMMK == 1
#if USE_BDA
		const __global real* restrict a_ptr = (const __global real* restrict) &agm[0];
		const __global real* restrict b_ptr = (const __global real* restrict) &bgm[0];
#else
		// use for scalar bgms
		int a_ptr_offset = a_offset*VWM;
		int b_ptr_offset = b_offset*VWN;
#endif
		const int tid_x = get_local_id(0) + MDIMC * GetGroupID0();
		const int tid_y = get_local_id(1) + NDIMC * GetGroupID1();
	#endif

	// Combined thread identifier (to disable caching)
	#if SA == 1 || SB == 1
		int tid = get_local_id(0) + MDIMC*get_local_id(1);
	#endif

	// Initializes the accumulation registers
	
	for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
		
		for (int _ni = 0; _ni < NWI; _ni += 1) {
			cpm[_ni * (MWI/VWM) + _mi] = InitAccRegisters();
		}
	}

	// Loops over all workgroup tiles
	for (int kwg = 0; kwg < kSizeK; kwg += KWG * KREG) {

		// Loads data: off-chip --> local (matrix A)
		#if SA == 1
			GlobalToLocalA(
#if USE_BDA
				agm,
#else
				a_offset,
#endif
				//alm,
				kSizeM, tid, kwg);
		#endif
		// Loads data: off-chip --> local (matrix B)
		#if SB == 1
			GlobalToLocalB(
#if USE_BDA
				bgm,
#else
				b_offset,
#endif
				//blm,
				kSizeN, tid, kwg);
		#endif
		#if SA == 1 || SB == 1
			barrier();
		#endif

		// Loops over all workitem tiles, unrolled by a factor KWI
		for (int pwi = 0; pwi < KWG * KREG; pwi += KWI * KREG) {
			
			for (int _pit = 0; _pit < KWI*KREG; _pit += KREG) {
				#if SA == 0 || SB == 0
					int idk = kwg + pwi + _pit;
				#endif
				#if SA == 1 || SB == 1
					int kg = pwi + _pit;
				#endif

				// Loads matrix A (kernel 0) or matrix B (kernel 1)
				
				for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
					// Loads data: local --> private (matrix A)
					#if GEMMK == 0 && SA == 1
						apm[_mi] = LocalToPrivateA(//alm,
							_mi, kg);
					// Loads data: off-chip --> private (matrix A)
					#elif GEMMK == 0 && SA == 0
						apm[_mi] = GlobalToPrivateA(
#if USE_BDA
							agm,
#else
							a_offset,
#endif
							_mi, kSizeM, idk, kwg);
					// Loads data: 2D global --> 2D private (matrix B)
					#elif GEMMK == 1
						
						for (int _ki = 0; _ki < KREG; _ki += 1) {
							bpm[_ki * (MWI/VWM) + _mi] = GlobalToPrivateB2D(
#if USE_BDA
								b_ptr,
#else
								b_ptr_offset,
#endif
								tid_x, _mi, kSizeN, idk, _ki);
						}
					#endif
				}

				// Loads matrix B (kernel 0) or matrix A (kernel 1)
				#if GEMMK == 0
					
					for (int _ni = 0; _ni < NWI/VWN; _ni += 1) {
						// Loads data: local --> private (matrix B)
						#if SB == 1
							bpm[_ni] = LocalToPrivateB(//blm,
								_ni, kg);
						// Loads data: off-chip --> private (matrix B)
						#else
							bpm[_ni] = GlobalToPrivateB(
#if USE_BDA
								bgm,
#else
								b_offset,
#endif
								_ni, kSizeN, idk);
						#endif
					}
				#elif GEMMK == 1
					// Loads data: 2D global --> 2D private (matrix A). Partly, shuffled later among subgroups
					#if USE_SUBGROUP_SHUFFLING == 1
						const int _ni = clblast_get_sub_group_local_id();
						
						for (int _ki = 0; _ki < KREG/VWN; _ki += 1) {
							apm[_ki] = GlobalToPrivateA2D(
#if USE_BDA
								a_ptr,
#else
								a_ptr_offset,
#endif
								tid_y, _ni, kSizeK, idk, _ki);
						}
					// Loads data: 2D global --> 2D private (matrix A)
					#else
						
						for (int _ni = 0; _ni < NWI; _ni += 1) {
							
							for (int _ki = 0; _ki < KREG/VWN; _ki += 1) {
								apm[_ni * (KREG/VWN) + _ki] = GlobalToPrivateA2D(
#if USE_BDA
									a_ptr,
#else
									a_ptr_offset,
#endif
									tid_y, _ni, kSizeK, idk, _ki);
							}
						}
					#endif
				#endif

				// Performs the accumulation (Cpm += Apm * Bpm)
				#if GEMMK == 0
					
					for (int _ni = 0; _ni < NWI/VWN; _ni += 1) {
						
						for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
							const realM aval = apm[_mi];
							#if VWN == 1
								cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi], aval, bpm[_ni]);
							#elif VWN == 2
								cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi], aval, bpm[_ni].x);
								cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi], aval, bpm[_ni].y);
							#elif VWN == 4
								cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi], aval, bpm[_ni].x);
								cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi], aval, bpm[_ni].y);
								cpm[(_ni*VWN + 2)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 2)*(MWI/VWM) + _mi], aval, bpm[_ni].z);
								cpm[(_ni*VWN + 3)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 3)*(MWI/VWM) + _mi], aval, bpm[_ni].w);
							#elif VWN == 8
								cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi], aval, bpm[_ni].s0);
								cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi], aval, bpm[_ni].s1);
								cpm[(_ni*VWN + 2)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 2)*(MWI/VWM) + _mi], aval, bpm[_ni].s2);
								cpm[(_ni*VWN + 3)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 3)*(MWI/VWM) + _mi], aval, bpm[_ni].s3);
								cpm[(_ni*VWN + 4)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 4)*(MWI/VWM) + _mi], aval, bpm[_ni].s4);
								cpm[(_ni*VWN + 5)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 5)*(MWI/VWM) + _mi], aval, bpm[_ni].s5);
								cpm[(_ni*VWN + 6)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 6)*(MWI/VWM) + _mi], aval, bpm[_ni].s6);
								cpm[(_ni*VWN + 7)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 7)*(MWI/VWM) + _mi], aval, bpm[_ni].s7);
							#elif VWN == 16
								cpm[(_ni*VWN + 0 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0 )*(MWI/VWM) + _mi], aval, bpm[_ni].s0);
								cpm[(_ni*VWN + 1 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 1 )*(MWI/VWM) + _mi], aval, bpm[_ni].s1);
								cpm[(_ni*VWN + 2 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 2 )*(MWI/VWM) + _mi], aval, bpm[_ni].s2);
								cpm[(_ni*VWN + 3 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 3 )*(MWI/VWM) + _mi], aval, bpm[_ni].s3);
								cpm[(_ni*VWN + 4 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 4 )*(MWI/VWM) + _mi], aval, bpm[_ni].s4);
								cpm[(_ni*VWN + 5 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 5 )*(MWI/VWM) + _mi], aval, bpm[_ni].s5);
								cpm[(_ni*VWN + 6 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 6 )*(MWI/VWM) + _mi], aval, bpm[_ni].s6);
								cpm[(_ni*VWN + 7 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 7 )*(MWI/VWM) + _mi], aval, bpm[_ni].s7);
								cpm[(_ni*VWN + 8 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 8 )*(MWI/VWM) + _mi], aval, bpm[_ni].s8);
								cpm[(_ni*VWN + 9 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 9 )*(MWI/VWM) + _mi], aval, bpm[_ni].s9);
								cpm[(_ni*VWN + 10)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 10)*(MWI/VWM) + _mi], aval, bpm[_ni].sA);
								cpm[(_ni*VWN + 11)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 11)*(MWI/VWM) + _mi], aval, bpm[_ni].sB);
								cpm[(_ni*VWN + 12)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 12)*(MWI/VWM) + _mi], aval, bpm[_ni].sC);
								cpm[(_ni*VWN + 13)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 13)*(MWI/VWM) + _mi], aval, bpm[_ni].sD);
								cpm[(_ni*VWN + 14)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 14)*(MWI/VWM) + _mi], aval, bpm[_ni].sE);
								cpm[(_ni*VWN + 15)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 15)*(MWI/VWM) + _mi], aval, bpm[_ni].sF);
							#endif
						}
					}
				#elif GEMMK == 1
					
					for (int _ni = 0; _ni < NWI; _ni += 1) {
						
						for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
							
							for (int _ki = 0; _ki < KREG/VWN; _ki += 1) {
								#if USE_SUBGROUP_SHUFFLING == 1
									const realN aval = clblast_sub_group_shuffle(apm[_ki], _ni);
								#else
									const realN aval = apm[_ni * (KREG/VWN) + _ki];
								#endif
								#if VWN == 1
									cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 0) * (MWI/VWM) + _mi], aval);
								#elif VWN == 2
									cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 0) * (MWI/VWM) + _mi], aval.x);
									cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 1) * (MWI/VWM) + _mi], aval.y);
								#elif VWN == 4
									cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 0) * (MWI/VWM) + _mi], aval.x);
									cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 1) * (MWI/VWM) + _mi], aval.y);
									cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 2) * (MWI/VWM) + _mi], aval.z);
									cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 3) * (MWI/VWM) + _mi], aval.w);
								#elif VWN == 8
									cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 0) * (MWI/VWM) + _mi], aval.s0);
									cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 1) * (MWI/VWM) + _mi], aval.s1);
									cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 2) * (MWI/VWM) + _mi], aval.s2);
									cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 3) * (MWI/VWM) + _mi], aval.s3);
									cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 4) * (MWI/VWM) + _mi], aval.s4);
									cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 5) * (MWI/VWM) + _mi], aval.s5);
									cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 6) * (MWI/VWM) + _mi], aval.s6);
									cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 7) * (MWI/VWM) + _mi], aval.s7);
								#elif VWN == 16
									cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 0 ) * (MWI/VWM) + _mi], aval.s0);
									cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 1 ) * (MWI/VWM) + _mi], aval.s1);
									cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 2 ) * (MWI/VWM) + _mi], aval.s2);
									cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 3 ) * (MWI/VWM) + _mi], aval.s3);
									cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 4 ) * (MWI/VWM) + _mi], aval.s4);
									cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 5 ) * (MWI/VWM) + _mi], aval.s5);
									cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 6 ) * (MWI/VWM) + _mi], aval.s6);
									cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 7 ) * (MWI/VWM) + _mi], aval.s7);
									cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 8 ) * (MWI/VWM) + _mi], aval.s8);
									cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 9 ) * (MWI/VWM) + _mi], aval.s9);
									cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 10) * (MWI/VWM) + _mi], aval.sA);
									cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 11) * (MWI/VWM) + _mi], aval.sB);
									cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 12) * (MWI/VWM) + _mi], aval.sC);
									cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 13) * (MWI/VWM) + _mi], aval.sD);
									cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 14) * (MWI/VWM) + _mi], aval.sE);
									cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 15) * (MWI/VWM) + _mi], aval.sF);
								#endif
							}
						}
					}
				#endif

			}
		}
		#if SA == 1 || SB == 1
			barrier();
		#endif
	}
	#if GLOBAL_MEM_FENCE == 1
		memoryBarrier(); barrier();
	#endif

	// Stores an MWG * NWG tile of results and performs the multiplication with alpha and beta
	#if GEMMK == 0
		const int cld = kSizeM;
	#elif GEMMK == 1
		const int cld = kSizeN;
	#endif
	
	for (int _ni = 0; _ni < NWI; _ni += 1) {
		
		for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
			StoreResults(
#if USE_BDA
				cgm,
#else
				c_offset,
#endif
				cpm[_ni * (MWI/VWM) + _mi], _mi, _ni, cld, alpha, beta);
		}
	}
}

//)"
// End of the C++11 raw string literal

// =================================================================================================
