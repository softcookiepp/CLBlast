
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains two optimized matrix-multiplication kernels:
// - Kernel 0: inspired by the paper by Matsumoto et al. and the tutorial on
//   http://www.cedricnugteren.nl/tutorial.php
// - Kernel 1: inspired by a Qualcomm optimized GPU kernel with 2D register tiling
//   https://developer.qualcomm.com/blog/matrix-multiply-adreno-gpus-part-2-host-code-and-kernel
// Both are fully configurable (and tunable!) using many parameters. Both kernels support
// different data-types (SGEMM/DGEMM/CGEMM/ZGEMM/HGEMM) through a pre-processor define.
//
// For kernel 0 matrices are accessed as follows:
// A: [k*M + m], with 'k' ranging from 0:K and 'm' from 0:M (m,k,m)
// B: [k*N + n], with 'k' ranging from 0:K and 'n' from 0:N (n,k,n)
// C: [n*M + m], with 'n' ranging from 0:N and 'm' from 0:M (m,n,m)
// For kernel 1, both A and C are transposed w.r.t. the above
//
// Or as an image (assuming column-major)
//       K                      
//    o-------o                 
//    |       |                 
//  N | [B^T] |                 
//    |       |                 
//    o-------o                 
//        K               N     
//    o-------o        o-----o  
//  M |  [A]  |      M | [C] |  
//    |       |        |     |  
//    o-------o        o-----o  
//                              
//
// This kernel is separated into multiple files. This is part 1 out of 4.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
//R"(
#ifndef XGEMM_PART1_GLSL
#define XGEMM_PART1_GLSL

#include "../common.glsl"
#include "level3.glsl"
// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library.

#ifndef USE_XGEMM_BATCHED
	#define USE_XGEMM_BATCHED 0
#endif

#if USE_XGEMM_BATCHED
	#define USE_SPECIALIZATION_CONSTANTS 0
#else
	#define USE_SPECIALIZATION_CONSTANTS 1
#endif

#if USE_SPECIALIZATION_CONSTANTS
	#ifdef GEMMK
		#undef GEMMK		
	#endif
	layout(constant_id = 0) const int GEMMK = 0; // Kernel to choose: 0 regular, 1 with 2D register tiling
	#ifdef MWG
		#undef MWG	
	#endif
	layout(constant_id = 1) const int MWG = 8; // Tile-size in dimension M (e.g. 64, 128)
	#ifdef NWG
		#undef NWG
	#endif
	layout(constant_id = 2) const int NWG = 8; // Tile-size in dimension N (e.g. 64, 128)
	#ifdef KWG
		#undef KWG
	#endif
	layout(constant_id = 3) const int KWG = 8; // Tile-size in dimension K (e.g. 8, 16)
	#ifdef MDIMC
		#undef MDIMC
	#endif
	layout(constant_id = 4) const int MDIMC = 8; // Threads per workgroup in M-dimension (e.g. 8, 16, 32)
	#ifdef NDIMC
		#undef NDIMC
	#endif
	layout(constant_id = 5) const int NDIMC = 8; // Threads per workgroup in N-dimension (e.g. 8, 16, 32)
	#ifdef MDIMA
		#undef MDIMA
	#endif
	layout(constant_id = 6) const int MDIMA = 8; // Re-shaped tile dimension of matrix A: KDIMA * MDIMA (kernel 0 only)
	#ifdef NDIMB
		#undef NDIMB
	#endif
	layout(constant_id = 7) const int NDIMB = 8; // Re-shaped tile dimension of matrix B: KDIMB * NDIMB (kernel 0 only)
	#ifdef KWI
		#undef KWI
	#endif
	layout(constant_id = 8) const int KWI = 1; // Unroll factor of the KWG loop (smaller or equal than KWG)
	#if 0 // this determines type, will not be possible to use as a specialization constant yet.
		#ifdef VWM
			#undef VWM
		#endif
		layout(constant_id = -1) const int VWM = 1; // Vector width of matrices A and C
		#ifdef VWN
			#undef VWN
		#endif
		layout(constant_id = -1) const int VWN = 1; // Vector width of matrix B
	#endif
	#ifdef STRM
		#undef STRM
	#endif
	layout(constant_id = 9) const int STRM = 0; // Use strided access within a thread in the M-dimension (1) or not (0) (kernel 0 only)
	#ifdef STRN
		#undef STRN
	#endif
	layout(constant_id = 10) const int STRN = 0; // Use strided access within a thread in the N-dimension (1) or not (0) (kernel 0 only)
	#ifdef SA
		#undef SA
	#endif
	layout(constant_id = 11) const int SA = 0; // Use local/shared memory to cache matrix A (1) or not (0) (kernel 0 only)
	#ifdef SB
		#undef SB
	#endif
	layout(constant_id = 12) const int SB = 0; // Use local/shared memory to cache matrix B (1) or not (0) (kernel 0 only)
	#ifdef KREG
		#undef KREG
	#endif
	layout(constant_id = 13) const int KREG = 1; // Amount of register tiling in second dimension, multiple of VWN (kernel 1 only)

	// Helper parameters based on the above tuning parameters
	#define MWI (MWG/MDIMC)							 // Work per work-item (M-dimension)
	#define NWI (NWG/NDIMC)							 // Work per work-item (N-dimension)
	#define KDIMA ((MDIMC*NDIMC)/(MDIMA)) // Re-shaped tile dimension of matrix A: KDIMA * MDIMA
	#define KDIMB ((MDIMC*NDIMC)/(NDIMB)) // Re-shaped tile dimension of matrix B: KDIMB * NDIMB
	#define MWA (MWG/MDIMA)							 // Amount of loads-per-thread for matrix A (M-dimension)
	#define KWA (KWG/KDIMA)							 // Amount of loads-per-thread for matrix A (K-dimension)
	#define KWB (KWG/KDIMB)							 // Amount of loads-per-thread for matrix B (K-dimension)
	#define NWB (NWG/NDIMB)							 // Amount of loads-per-thread for matrix B (N-dimension)
	
	// Settings
	#ifdef USE_VECTOR_MAD
		#undef USE_VECTOR_MAD 0			// Unroll (0) or don't (1) unroll the vector MAD manually
	#endif
	layout(constant_id = 14) const int USE_VECTOR_MAD = 0; // Unroll (0) or don't (1) unroll the vector MAD manually
	
	// this logic doesn't apply if subgroup operations aren't supported at all
	#if SUBGROUP_OPERATIONS_SUPPORTED == 1
		#ifdef USE_SUBGROUP_SHUFFLING
			#undef USE_SUBGROUP_SHUFFLING	 
		#endif
		layout(constant_id = 15) const int USE_SUBGROUP_SHUFFLING = 0; // Optionally enables subgroup shuffling for supported GPUs
	#endif
#else
	#ifndef GEMMK
		#define GEMMK 0		// Kernel to choose: 0 regular, 1 with 2D register tiling
	#endif
	#ifndef MWG
		#define MWG 8			// Tile-size in dimension M (e.g. 64, 128)
	#endif
	#ifndef NWG
		#define NWG 8			// Tile-size in dimension N (e.g. 64, 128)
	#endif
	#ifndef KWG
		#define KWG 8			// Tile-size in dimension K (e.g. 8, 16)
	#endif
	#ifndef MDIMC
		#define MDIMC 8		// Threads per workgroup in M-dimension (e.g. 8, 16, 32)
	#endif
	#ifndef NDIMC
		#define NDIMC 8		// Threads per workgroup in N-dimension (e.g. 8, 16, 32)
	#endif
	#ifndef MDIMA
		#define MDIMA 8		// Re-shaped tile dimension of matrix A: KDIMA * MDIMA (kernel 0 only)
	#endif
	#ifndef NDIMB
		#define NDIMB 8		// Re-shaped tile dimension of matrix B: KDIMB * NDIMB (kernel 0 only)
	#endif
	#ifndef KWI
		#define KWI 1			// Unroll factor of the KWG loop (smaller or equal than KWG)
	#endif
	#ifndef VWM
		#define VWM 1			// Vector width of matrices A and C
	#endif
	#ifndef VWN
		#define VWN 1			// Vector width of matrix B
	#endif
	#ifndef STRM
		#define STRM 0		 // Use strided access within a thread in the M-dimension (1) or not (0) (kernel 0 only)
	#endif
	#ifndef STRN
		#define STRN 0		 // Use strided access within a thread in the N-dimension (1) or not (0) (kernel 0 only)
	#endif
	#ifndef SA
		#define SA 0			 // Use local/shared memory to cache matrix A (1) or not (0) (kernel 0 only)
	#endif
	#ifndef SB
		#define SB 0			 // Use local/shared memory to cache matrix B (1) or not (0) (kernel 0 only)
	#endif
	#ifndef KREG
		#define KREG 1		 // Amount of register tiling in second dimension, multiple of VWN (kernel 1 only)
	#endif

	// Helper parameters based on the above tuning parameters
	#define MWI (MWG/MDIMC)							 // Work per work-item (M-dimension)
	#define NWI (NWG/NDIMC)							 // Work per work-item (N-dimension)
	#define KDIMA ((MDIMC*NDIMC)/(MDIMA)) // Re-shaped tile dimension of matrix A: KDIMA * MDIMA
	#define KDIMB ((MDIMC*NDIMC)/(NDIMB)) // Re-shaped tile dimension of matrix B: KDIMB * NDIMB
	#define MWA (MWG/MDIMA)							 // Amount of loads-per-thread for matrix A (M-dimension)
	#define KWA (KWG/KDIMA)							 // Amount of loads-per-thread for matrix A (K-dimension)
	#define KWB (KWG/KDIMB)							 // Amount of loads-per-thread for matrix B (K-dimension)
	#define NWB (NWG/NDIMB)							 // Amount of loads-per-thread for matrix B (N-dimension)

	// Settings
	#ifndef USE_VECTOR_MAD
		#define USE_VECTOR_MAD 0			// Unroll (0) or don't (1) unroll the vector MAD manually
	#endif

	// this logic doesn't apply if subgroup operations aren't supported at all
	#if SUBGROUP_OPERATIONS_SUPPORTED == 1
		#ifndef USE_SUBGROUP_SHUFFLING
			#define USE_SUBGROUP_SHUFFLING 0		 // Optionally enables subgroup shuffling for Intel GPUs
		#endif
	#endif
#endif

// =================================================================================================

// Data-widths in dimension M
#if VWM == 1
		#define realM real
#elif VWM == 2
		#define realM real2
#elif VWM == 4
		#define realM real4
#elif VWM == 8
		#define realM real8
#elif VWM == 16
		#define realM real16
#endif

// Data-widths in dimension N
#if VWN == 1
		#define realN real
#elif VWN == 2
		#define realN real2
#elif VWN == 4
		#define realN real4
#elif VWN == 8
		#define realN real8
#elif VWN == 16
		#define realN real16
#endif

// =================================================================================================

// Initializes the accumulation registers to zero
realM InitAccRegisters()
{
	realM result;
	#if VWM == 1
		SetToZero(result);
	#else
		vSetToZero(result, VWM);
	#endif
	return result;
}

// =================================================================================================

// buffer definitions (to avoid having to use macros everywhere like usual)
#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer agm_buf { realM agm[]; };
	layout(binding = 1, std430) readonly buffer bgm_buf { realN bgm[]; };
	layout(binding = 2, std430) buffer cgm_buf { realM cgm[]; };
	layout(binding = 3, std430) readonly buffer agms_buf { real a_ptr[]; };
	layout(binding = 4, std430) readonly buffer bgms_buf { real b_ptr[]; };
	#if USE_XGEMM_BATCHED == 1
		layout(binding = 5, std430) readonly buffer arg_alphas_buf { real_arg arg_alphas[]; };
		layout(binding = 6, std430) readonly buffer arg_betas_buf { real_arg arg_betas[]; };
	#endif
#endif

// Allocates workgroup-private memory (local memory)
// Not always used, but is sometimes
shared realM alm[KWG * MWG/VWM];
shared realN blm[KWG * NWG/VWN];

// =================================================================================================

// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for
// caching the A input matrix.
//#if SA == 1
void GlobalToLocalA(
	#if USE_BDA
		const __global realM* restrict agm,
	#else
		int a_offset,
	#endif
	//LOCAL_PTR realM* alm,
	const int kSizeM, const int tid, const int kwg)
{
	const int la0 = tid % MDIMA;
	const int la1 = tid / MDIMA;
	
	for (int _mia = 0; _mia < MWA/VWM; _mia += 1)
	{
		for (int _kia = 0; _kia < KWA; _kia += 1)
		{
			// Computes the indices based on strided/non-strided access
			int mg;
			if (STRM == 0)
				mg = _mia + la0*(MWA/VWM);
			else if (STRM == 1)
				mg = la0 + _mia*MDIMA;

			// Computes the indices for the global memory
			int kg = _kia + la1*KWA;
			int idm = mg + GetGroupID0() * (MWG/VWM);
			int idk = kg + kwg;

			// Loads the data from global memory (not transposed) into the local memory
			alm[kg*(MWG/VWM) + mg] = agm[idk*(kSizeM/VWM) + idm + a_offset];
		}
	}
}

// Same as above, but now for the B input matrix
//#if SB == 1
void GlobalToLocalB(
	#if USE_BDA
		const __global realN* restrict bgm,
	#else
		int b_offset,
	#endif
	// LOCAL_PTR realN* blm,
	const int kSizeN, const int tid, const int kwg)
{
	const int lb0 = tid % NDIMB;
	const int lb1 = tid / NDIMB;
	
	for (int _kib = 0; _kib < KWB; _kib += 1) {
		
		for (int _nib = 0; _nib < NWB/VWN; _nib += 1) {

			// Computes the indices based on strided/non-strided access
			int ng;
			if (STRN == 0) ng = _nib + lb0*(NWB/VWN);
			else if (STRN == 1) ng = lb0 + _nib*NDIMB;

			// Computes the indices for the global memory
			int kg = _kib + lb1*KWB;
			int idn = ng + GetGroupID1() * (NWG/VWN);
			int idk = kg + kwg;

			// Loads the data from global memory (transposed) into the local memory
			blm[kg*(NWG/VWN) + ng] = bgm[idk*(kSizeN/VWN) + idn + b_offset];
		}
	}
}

// =================================================================================================

// Caches global off-chip memory directly into per-thread private memory (registers). This function
// is specific for caching the A input matrix.
//#if SA == 0 && GEMMK == 0
realM GlobalToPrivateA(
	#if USE_BDA
		const __global realM* restrict agm,
	#else
		int a_offset,
	#endif
	const int _mi, const int kSizeM, const int idk, const int kwg)
{
	// Computes the indices based on strided/non-strided access
	int mg;
	if (STRM == 0) mg = _mi + get_local_id(0)*(MWI/VWM);
	else if (STRM == 1) mg = get_local_id(0) + _mi*MDIMC;

	// Computes the indices for the global memory
	int idm = mg + GetGroupID0() * (MWG/VWM);

	// Loads the data from global memory (not transposed) and stores into registers
	return agm[idk*(kSizeM/VWM) + idm + a_offset];
}


// Same as above, but now for the B input matrix
//#if SB == 0 && GEMMK == 0
realN GlobalToPrivateB(
	#if USE_BDA
		const __global realN* restrict bgm,
	#else
		int b_offset,
	#endif
	const int _ni, const int kSizeN, const int idk)
{
	// Computes the indices based on strided/non-strided access
	int ng;
	if (STRN == 0) ng = _ni + get_local_id(1)*(NWI/VWN);
	else if (STRN == 1) ng = get_local_id(1) + _ni*NDIMC;

	// Computes the indices for the global memory
	int idn = ng + GetGroupID1() * (NWG/VWN);

	// Loads the data from global memory (transposed) and stores into registers
	return bgm[idk*(kSizeN/VWN) + idn + b_offset];
}

// =================================================================================================

// Caches global off-chip memory directly into per-thread private memory (registers). This function
// is specific for caching the A input matrix for kernel 1.
realN GlobalToPrivateA2D(
	#if USE_BDA
		const __global real* restrict a_ptr,
	#else
		int a_ptr_offset,
	#endif
	const int tid_y, const int _ni, const int kSizeK, const int idk, const int _ki)
{
	#if ROUTINE_IS_COMPLEX
		const int a_index = (tid_y * NWI + _ni) * (kSizeK / VWN) + idk / VWN + _ki;
		#if USE_BDA
			const __global realN* restrict agm = (const __global realN* restrict) a_ptr;
		#endif
		// ok yeah, this is probably not going to work quite the way I thought it would...
		return agm[a_index];
	#else
		const int a_index = (tid_y * NWI + _ni) * kSizeK + idk + _ki * VWN + a_ptr_offset;
		#if VWN == 1
			return a_ptr[a_index];
		#else
			//return vload2(0, a_ptr + a_index);
			realN outp;
			vloadN2(outp, a_index, a_ptr, VWN);
			return outp;
		#endif
	#endif
}

// Same as above, but now for the B input matrix
realM GlobalToPrivateB2D(
	#if USE_BDA
		const __global real* restrict b_ptr,
	#else
		int b_ptr_offset,
	#endif
	const int tid_x, const int _mi, const int kSizeN, const int idk, const int _ki)
{
	#if ROUTINE_IS_COMPLEX
		const int b_index = (idk + _ki) * (kSizeN / VWM) + tid_x * (MWI / VWM) + _mi;
		#if USE_BDA
			const __global realM* restrict bgm = (const __global realM* restrict) b_ptr;
		#endif
		// ok yeah, this is probably not going to work quite the way I thought it would...
		return bgm[b_index + b_ptr_offset/VWM];
	#else
		const int b_index = (idk + _ki) * kSizeN + tid_x * MWI + _mi * VWM + b_ptr_offset;
		#if VWM == 1
			return b_ptr[b_index];
		#elif 1
			realM ret;
			vloadN2(ret, b_index, b_ptr, VWM);
			return ret;
		#else
			#if VWM == 2
				//return vload2(0, b_ptr + b_index);
				return real2(b_ptr[b_index], b_ptr[b_index + 1]);
			#elif VWM == 4
				//return vload4(0, b_ptr + b_index);
				return real4(
					b_ptr[b_index],
					b_ptr[b_index + 1]
					b_ptr[b_index + 2]
					b_ptr[b_index + 3]
				);
			#elif VWM == 8
				//return vload8(0, b_ptr + b_index);
				return real8(
					real4(
						b_ptr[b_index],
						b_ptr[b_index + 1]
						b_ptr[b_index + 2]
						b_ptr[b_index + 3]
					),
					real4(
						b_ptr[b_index + 4],
						b_ptr[b_index + 5]
						b_ptr[b_index + 6]
						b_ptr[b_index + 7]
					)
				);
			#elif VWM == 16
				//return vload16(0, b_ptr + b_index);
				return real16(
					real4(
						b_ptr[b_index],
						b_ptr[b_index + 1]
						b_ptr[b_index + 2]
						b_ptr[b_index + 3]
					),
					real4(
						b_ptr[b_index + 4],
						b_ptr[b_index + 5]
						b_ptr[b_index + 6]
						b_ptr[b_index + 7]
					),
					real4(
						b_ptr[b_index + 8],
						b_ptr[b_index + 9]
						b_ptr[b_index + 10]
						b_ptr[b_index + 11]
					),
					real4(
						b_ptr[b_index + 12],
						b_ptr[b_index + 13]
						b_ptr[b_index + 14]
						b_ptr[b_index + 15]
					)
				);
			#endif
		#endif
	#endif
}

// =================================================================================================
//pp
// Caches on-chip local memory into per-thread private memory (registers). This function is specific
// for caching the A input matrix.
realM LocalToPrivateA(
	//LOCAL_PTR realM* alm,
	const int _mi, const int kg)
{
	int mg;
	if (STRM == 0) mg = _mi + get_local_id(0)*(MWI/VWM);
	else if (STRM == 1) mg = get_local_id(0) + _mi*MDIMC;
	return alm[kg*(MWG/VWM) + mg];
}

// Same as above, but now for the B input matrix
realN LocalToPrivateB(
	//LOCAL_PTR realN* blm,
	const int _ni, const int kg)
{
	int ng;
	if (STRN == 0) ng = _ni + get_local_id(1)*(NWI/VWN);
	else if (STRN == 1) ng = get_local_id(1) + _ni*NDIMC;
	return blm[kg*(NWG/VWN) + ng];
}
#endif
// End of the C++11 raw string literal
//)"
// =================================================================================================
