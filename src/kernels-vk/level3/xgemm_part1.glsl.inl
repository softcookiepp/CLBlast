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

// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library.
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
#ifndef GLOBAL_MEM_FENCE
	#define GLOBAL_MEM_FENCE 0		// Global synchronisation barrier for potential better performance
#endif

#ifndef SUBGROUP_SHUFFLING_NVIDIA_PRE_VOLTA
	#define SUBGROUP_SHUFFLING_NVIDIA_PRE_VOLTA 0
#endif
#ifndef SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA
	#define SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA 0
#endif
#ifndef SUBGROUP_SHUFFLING_INTEL
	#define SUBGROUP_SHUFFLING_INTEL 0
#endif
#ifndef USE_SUBGROUP_SHUFFLING
	#define USE_SUBGROUP_SHUFFLING 0		 // Optionally enables subgroup shuffling for Intel GPUs
#endif

// Intel subgroups (https://www.khronos.org/registry/OpenCL/extensions/intel/cl_intel_subgroups.html)
#if USE_SUBGROUP_SHUFFLING == 1 && SUBGROUP_SHUFFLING_INTEL == 1
	#pragma OPENCL EXTENSION cl_intel_subgroups: enable
	#define SUBGROUP_SIZE 8							// Assumes subgroup size is always 8 on Intel GPUs
#endif

// NVIDIA warps as subgroups using inline PTX (https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html)
#if USE_SUBGROUP_SHUFFLING == 1
	#if SUBGROUP_SHUFFLING_NVIDIA_PRE_VOLTA == 1 || SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA == 1
		#define SUBGROUP_SIZE 32						// Assumes subgroup size is always 32 on NVIDIA GPUs
	#endif
#endif

#if NWI != SUBGROUP_SIZE || MDIMC < SUBGROUP_SIZE
	#undef USE_SUBGROUP_SHUFFLING
	#define USE_SUBGROUP_SHUFFLING 0		 // Disables subgroups in case the assumptions don't hold
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
INLINE_FUNC realM InitAccRegisters() {
	realM result;
	#if VWM == 1
		SetToZero(result);
	#elif VWM == 2
		SetToZero(result.x);
		SetToZero(result.y);
	#elif VWM == 4
		SetToZero(result.x);
		SetToZero(result.y);
		SetToZero(result.z);
		SetToZero(result.w);
	#elif VWM == 8
		// TODO: implement expressions for this and VW == 16
		SetToZero(result.s0);
		SetToZero(result.s1);
		SetToZero(result.s2);
		SetToZero(result.s3);
		SetToZero(result.s4);
		SetToZero(result.s5);
		SetToZero(result.s6);
		SetToZero(result.s7);
	#elif VWM == 16
		SetToZero(result.s0);
		SetToZero(result.s1);
		SetToZero(result.s2);
		SetToZero(result.s3);
		SetToZero(result.s4);
		SetToZero(result.s5);
		SetToZero(result.s6);
		SetToZero(result.s7);
		SetToZero(result.s8);
		SetToZero(result.s9);
		SetToZero(result.sA);
		SetToZero(result.sB);
		SetToZero(result.sC);
		SetToZero(result.sD);
		SetToZero(result.sE);
		SetToZero(result.sF);
	#endif
	return result;
}

// =================================================================================================

// buffer definitions (to avoid having to use macros everywhere like usual)
#if USE_BDA == 0
	layout(binding = 0, std430) buffer agm_buf { realM agm[]; };
	layout(binding = 1, std430) buffer bgm_buf { realN bgm[]; };
	layout(binding = 2, std430) buffer cgm_buf { realM cgm[]; };
	#if GEMMK == 1
		layout(binding = 3, std430) buffer agms_buf { real a_ptr[]; };
		layout(binding = 4, std430) buffer bgms_buf { real b_ptr[]; };
	#endif
#endif

// Allocates workgroup-private memory (local memory)
#if SA == 1
	shared realM alm[KWG * MWG/VWM];
#endif
#if SB == 1
	shared realN blm[KWG * NWG/VWN];
#endif

// =================================================================================================

// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for
// caching the A input matrix.
#if SA == 1
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
			#if STRM == 0
				int mg = _mia + la0*(MWA/VWM);
			#elif STRM == 1
				int mg = la0 + _mia*MDIMA;
			#endif

			// Computes the indices for the global memory
			int kg = _kia + la1*KWA;
			int idm = mg + GetGroupID0() * (MWG/VWM);
			int idk = kg + kwg;

			// Loads the data from global memory (not transposed) into the local memory
			alm[kg*(MWG/VWM) + mg] = agm[idk*(kSizeM/VWM) + idm + a_offset];
		}
	}
}
#endif

// Same as above, but now for the B input matrix
#if SB == 1
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
			#if STRN == 0
				int ng = _nib + lb0*(NWB/VWN);
			#elif STRN == 1
				int ng = lb0 + _nib*NDIMB;
			#endif

			// Computes the indices for the global memory
			int kg = _kib + lb1*KWB;
			int idn = ng + GetGroupID1() * (NWG/VWN);
			int idk = kg + kwg;

			// Loads the data from global memory (transposed) into the local memory
			blm[kg*(NWG/VWN) + ng] = bgm[idk*(kSizeN/VWN) + idn + b_offset];
		}
	}
}
#endif

// =================================================================================================

// Caches global off-chip memory directly into per-thread private memory (registers). This function
// is specific for caching the A input matrix.
#if SA == 0 && GEMMK == 0
realM GlobalToPrivateA(
#if USE_BDA
	const __global realM* restrict agm,
#else
	int a_offset,
#endif
	const int _mi, const int kSizeM, const int idk, const int kwg)
{
	// Computes the indices based on strided/non-strided access
	#if STRM == 0
		int mg = _mi + get_local_id(0)*(MWI/VWM);
	#elif STRM == 1
		int mg = get_local_id(0) + _mi*MDIMC;
	#endif

	// Computes the indices for the global memory
	int idm = mg + GetGroupID0() * (MWG/VWM);

	// Loads the data from global memory (not transposed) and stores into registers
	return agm[idk*(kSizeM/VWM) + idm + a_offset];
}
#endif

// Same as above, but now for the B input matrix
#if SB == 0 && GEMMK == 0
realN GlobalToPrivateB(
#if USE_BDA
	const __global realN* restrict bgm,
#else
	int b_offset,
#endif
	const int _ni, const int kSizeN, const int idk)
{
	// Computes the indices based on strided/non-strided access
	#if STRN == 0
		int ng = _ni + get_local_id(1)*(NWI/VWN);
	#elif STRN == 1
		int ng = get_local_id(1) + _ni*NDIMC;
	#endif

	// Computes the indices for the global memory
	int idn = ng + GetGroupID1() * (NWG/VWN);

	// Loads the data from global memory (transposed) and stores into registers
	return bgm[idk*(kSizeN/VWN) + idn + b_offset];
}
#endif

// =================================================================================================
#if GEMMK == 1

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
	#if PRECISION == 3232 || PRECISION == 6464
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
		#elif VWN == 2
			//return vload2(0, a_ptr + a_index);
			return real2(a_ptr[a_index], a_ptr[a_index + 1]);
		#elif VWN == 4
			//return vload4(0, a_ptr + a_index);
			return real4(
				a_ptr[a_index],
				a_ptr[a_index + 1]
				a_ptr[a_index + 2]
				a_ptr[a_index + 3]
			);
		#elif VWN == 8
			//return vload8(0, a_ptr + a_index);
			return real8(
				real4(
					a_ptr[a_index],
					a_ptr[a_index + 1]
					a_ptr[a_index + 2]
					a_ptr[a_index + 3]
				),
				real4(
					a_ptr[a_index + 4],
					a_ptr[a_index + 5]
					a_ptr[a_index + 6]
					a_ptr[a_index + 7]
				)
			);
		#elif VWN == 16
			//return vload16(0, a_ptr + a_index);
			return real16(
				real4(
					a_ptr[a_index],
					a_ptr[a_index + 1]
					a_ptr[a_index + 2]
					a_ptr[a_index + 3]
				),
				real4(
					a_ptr[a_index + 4],
					a_ptr[a_index + 5]
					a_ptr[a_index + 6]
					a_ptr[a_index + 7]
				),
				real4(
					a_ptr[a_index + 8],
					a_ptr[a_index + 9]
					a_ptr[a_index + 10]
					a_ptr[a_index + 11]
				),
				real4(
					a_ptr[a_index + 12],
					a_ptr[a_index + 13]
					a_ptr[a_index + 14]
					a_ptr[a_index + 15]
				)
			);
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
	#if PRECISION == 3232 || PRECISION == 6464
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
		#elif VWM == 2
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
}

#endif
// =================================================================================================

// Caches on-chip local memory into per-thread private memory (registers). This function is specific
// for caching the A input matrix.
#if SA == 1
realM LocalToPrivateA(
	//LOCAL_PTR realM* alm,
	const int _mi, const int kg)
{
	#if STRM == 0
		int mg = _mi + get_local_id(0)*(MWI/VWM);
	#elif STRM == 1
		int mg = get_local_id(0) + _mi*MDIMC;
	#endif
	return alm[kg*(MWG/VWM) + mg];
}
#endif

// Same as above, but now for the B input matrix
#if SB == 1
realN LocalToPrivateB(
	//LOCAL_PTR realN* blm,
	const int _ni, const int kg)
{
	#if STRN == 0
		int ng = _ni + get_local_id(1)*(NWI/VWN);
	#elif STRN == 1
		int ng = get_local_id(1) + _ni*NDIMC;
	#endif
	return blm[kg*(NWG/VWN) + ng];
}
#endif

// End of the C++11 raw string literal
)"
// =================================================================================================
