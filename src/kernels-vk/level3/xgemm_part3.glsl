
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
#ifndef XGEMM_PART3_GLSL
#define XGEMM_PART3_GLSL

#include "xgemm_part2.glsl"
// A common interface for subgroup functions
// genuinely no idea how this maps to GLSL as of now; Vulkan probably has entirely different extensions
// We will just have to disable it host-side until a solution is found...
#if USE_SUBGROUP_SHUFFLING == 1

int clblast_get_sub_group_local_id()
{
	return get_sub_group_local_id();
}

realN clblast_sub_group_shuffle(realN reg, int src)
{
	return subgroupShuffle(reg, uint(src));
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
	// Different register variables for different GEMMK ids. Will be a testament to the power of specialization constants later on.
	// GEMMK == 0
	realM apm_gk0[MWI/VWM]; // MWI * 1
	realN bpm_gk0[NWI/VWN]; // 1 * NWI
	// GEMMK == 1
	#if USE_SUBGROUP_SHUFFLING == 1
		realN apm_gk1[KREG/VWN]; // KREG (subgroup shuffling in NWI dimension)
	#else
		realN apm_gk1[NWI*(KREG/VWN)]; // NWI * KREG
	#endif
	realM bpm_gk1[KREG*(MWI/VWM)]; // KREG * MWI
	
	realM cpm[NWI*(MWI/VWM)]; // NWI * MWI
	
	int tid_x, tid_y, tid;
	int a_ptr_offset, b_ptr_offset;
	if (GEMMK == 1)
	{
		#if USE_BDA
			const __global real* restrict a_ptr = (const __global real* restrict) &agm[0];
			const __global real* restrict b_ptr = (const __global real* restrict) &bgm[0];
		#else
			// use for scalar bgms
			a_ptr_offset = a_offset*VWM;
			b_ptr_offset = b_offset*VWN;
		#endif
		tid_x = get_local_id(0) + MDIMC * GetGroupID0();
		tid_y = get_local_id(1) + NDIMC * GetGroupID1();
	}

	// Combined thread identifier (to disable caching)
	if (SA == 1 || SB == 1)
		tid = get_local_id(0) + MDIMC*get_local_id(1);

	// Initializes the accumulation registers
	
	for (int _mi = 0; _mi < MWI/VWM; _mi += 1)
	{	
		for (int _ni = 0; _ni < NWI; _ni += 1)
		{
			cpm[_ni * (MWI/VWM) + _mi] = InitAccRegisters();
		}
	}

	// Loops over all workgroup tiles
	[[unroll]]
	for (int kwg = 0; kwg < kSizeK; kwg += KWG * KREG)
	{
		// Loads data: off-chip --> local (matrix A)
		if (SA == 1)
			GlobalToLocalA(
				#if USE_BDA
					agm,
				#else
					a_offset,
				#endif
				//alm,
				kSizeM, tid, kwg);

		// Loads data: off-chip --> local (matrix B)
		if (SB == 1)
			GlobalToLocalB(
				#if USE_BDA
					bgm,
				#else
					b_offset,
				#endif
				//blm,
				kSizeN, tid, kwg);
		
		if (SA == 1 || SB == 1)
			barrier();

		// Loops over all workitem tiles, unrolled by a factor KWI
		for (int pwi = 0; pwi < KWG * KREG; pwi += KWI * KREG) {
			
			for (int _pit = 0; _pit < KWI*KREG; _pit += KREG)
			{
				int idk;
				int kg;
				if (SA == 0 || SB == 0)
					idk = kwg + pwi + _pit;
				if (SA == 1 || SB == 1)
					kg = pwi + _pit;

				// Loads matrix A (kernel 0) or matrix B (kernel 1)
				
				for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
					// Loads data: local --> private (matrix A)
					if (GEMMK == 0 && SA == 1)
					{
						apm_gk0[_mi] = LocalToPrivateA(//alm,
							_mi, kg);
					}
					// Loads data: off-chip --> private (matrix A)
					else if (GEMMK == 0 && SA == 0)
					{
						apm_gk0[_mi] = GlobalToPrivateA(
							#if USE_BDA
								agm,
							#else
								a_offset,
							#endif
							_mi, kSizeM, idk, kwg);
					}
					// Loads data: 2D global --> 2D private (matrix B)
					else if (GEMMK == 1)
					{
						for (int _ki = 0; _ki < KREG; _ki += 1)
						{
							bpm_gk1[_ki * (MWI/VWM) + _mi] = GlobalToPrivateB2D(
								#if USE_BDA
									b_ptr,
								#else
									b_ptr_offset,
								#endif
								tid_x, _mi, kSizeN, idk, _ki);
						}
					}
				}

				// Loads matrix B (kernel 0) or matrix A (kernel 1)
				if (GEMMK == 0)
				{
					for (int _ni = 0; _ni < NWI/VWN; _ni += 1)
					{
						// Loads data: local --> private (matrix B)
						if (SB == 1)
							bpm_gk0[_ni] = LocalToPrivateB(//blm,
								_ni, kg);
						// Loads data: off-chip --> private (matrix B)
						else
							bpm_gk0[_ni] = GlobalToPrivateB(
								#if USE_BDA
									bgm,
								#else
									b_offset,
								#endif
								_ni, kSizeN, idk);
					}
				}
				else if (GEMMK == 1)
				{
					// Loads data: 2D global --> 2D private (matrix A). Partly, shuffled later among subgroups
					#if USE_SUBGROUP_SHUFFLING == 1
						const int _ni = clblast_get_sub_group_local_id();
						
						for (int _ki = 0; _ki < KREG/VWN; _ki += 1) {
							apm_gk1[_ki] = GlobalToPrivateA2D(
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
								apm_gk1[_ni * (KREG/VWN) + _ki] = GlobalToPrivateA2D(
									#if USE_BDA
										a_ptr,
									#else
										a_ptr_offset,
									#endif
									tid_y, _ni, kSizeK, idk, _ki);
							}
						}
					#endif
				}

				// Performs the accumulation (Cpm += Apm * Bpm)
				if (GEMMK == 0)
					
					for (int _ni = 0; _ni < NWI/VWN; _ni += 1) {
						
						for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
							const realM aval = apm_gk0[_mi];
							#if VWN == 1
								cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi], aval, bpm_gk0[_ni]);
							#else
								UNROLL(VWN)
								for (uint iv = 0; iv < VWN; iv += 1)
									cpm[(_ni*VWN + iv )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + iv )*(MWI/VWM) + _mi], aval, bpm_gk0[_ni].s[iv]);
							#endif
						}
					}
				else if (GEMMK == 1)
				{
					for (int _ni = 0; _ni < NWI; _ni += 1) {
						
						for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
							
							for (int _ki = 0; _ki < KREG/VWN; _ki += 1) {
								#if USE_SUBGROUP_SHUFFLING == 1
									const realN aval = clblast_sub_group_shuffle(apm_gk1[_ki], _ni);
								#else
									const realN aval = apm_gk1[_ni * (KREG/VWN) + _ki];
								#endif
								#if VWN == 1
									cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm_gk1[(VWN * _ki + 0) * (MWI/VWM) + _mi], aval);
								#else
									UNROLL(VWN)
									for (uint iv = 0; iv < VWN; iv += 1)
										cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm_gk1[(VWN * _ki + iv) * (MWI/VWM) + _mi], aval.s[iv]);
								#endif
							}
						}
					}
				}
			}
		}
		if (SA == 1 || SB == 1)
			barrier();
	}

	// Stores an MWG * NWG tile of results and performs the multiplication with alpha and beta
	int cld = kSizeM;
	if (GEMMK == 0)
		cld = kSizeM;
	else if (GEMMK == 1)
		cld = kSizeN;
	
	for (int _ni = 0; _ni < NWI; _ni += 1)
	{
		
		for (int _mi = 0; _mi < MWI/VWM; _mi += 1)
		{
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
#endif
//)"
// End of the C++11 raw string literal

// =================================================================================================
