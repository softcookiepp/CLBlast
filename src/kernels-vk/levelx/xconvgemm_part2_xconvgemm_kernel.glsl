#version 450
#include "../level3/xgemm_direct_part2.glsl"
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the an implementation of 3D convolution on a 4D image using GEMM kernels. It
// uses parameters from the direct GEMM kernel. This part contains the main kernel (2/2).
// This uses "CONVGEMM_WITH_IM2COL" as a switch to select between direct convgemm or first running
// the im2col kernel to create a 'col' temporary matrix.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
//R"(

// =================================================================================================

// ConvGEMM kernel
#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = MDIMCD, local_size_y = NDIMCD, local_size_z = 1) in;
#endif

#if USE_BDA == 0
	layout(binding = 0, std430) buffer kernelgm_buffer { realND kernelgm[]; };
	layout(binding = 1, std430) buffer resultgm_buffer { real resultgm[]; };
	layout(binding = 2, std430) buffer colgm_buffer { realMD colgm[]; };
	// workaround for no pointer casting
	layout(binding = 3, std430) buffer kernelgms_buffer { real kernelgms[]; };
	layout(binding = 4, std430) buffer colgms_buffer { real colgms[]; };
#endif

layout(push_constant, std430) uniform Xconvgemm
{
	int num_patches; int num_kernels; int patch_size;
#if USE_BDA
	__global realND* restrict kernelgm;
#endif
	int kernel_offset;
#if USE_BDA
	__global real* resultgm;
#endif
	int result_offset; int result_stride;
#if USE_BDA
	__global realMD* restrict colgm;
#endif
	int col_offset; int col_stride;
} args;

// these may be defined elsewhere. I really don't know at this point
shared real alm[WGD * (WGD + PADA)];
shared real blm[WGD * (WGD + PADB)];

void main()
{
	// Batch offsets
	const int batch = get_group_id(2);
	const int col_offset_batch = args.col_offset + args.col_stride * batch;
	const int result_offset_batch = args.result_offset + args.result_stride * batch;

	// Extra pointers to scalar versions of global memory
#if USE_BDA
	const __global real* restrict colgms = (const __global real* restrict) colgm;
	const __global real* restrict kernelgms = (const __global real* restrict) kernelgm;
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

	// Global m/n indices
	const int idm = get_local_id(0) * MWID + GetGroupID0() * WGD;
	const int idn = get_local_id(1) * NWID + GetGroupID1() * WGD;

	// The faster version of GEMM is not allowed on the (incomplete) borders. Therefore, this section
	// processes only the main parts: output blocks of WGD by WGD.
	if ((idm < (args.num_patches/WGD)*WGD) && (idn < (args.num_kernels/WGD)*WGD)) {

		// Loops over all complete workgroup tiles (K-dimension)
		int kwg = 0;
		for (kwg = 0; kwg < (args.patch_size/WGD) * WGD; kwg += WGD) {

			// Loads data: off-chip --> local (matrix A and B)
			if (args.num_patches % VWMD == 0 && col_offset_batch % VWMD == 0) {
				GlobalToLocalDirectA(colgm, alm, args.num_patches, col_offset_batch, kwg, false, false);
			}
			else {
				GlobalToLocalScalarA(colgms, alm, args.num_patches, col_offset_batch, kwg, false, false);
			}
			
			if (args.patch_size % VWND == 0 && args.kernel_offset % VWND == 0) {
				GlobalToLocalDirectB(kernelgm, blm, args.patch_size, args.kernel_offset, kwg, true, false);
			}
			else {
				GlobalToLocalScalarB(kernelgms, blm, args.patch_size, args.kernel_offset, kwg, true, false);
			}
			barrier();

			// Loops over all workitem tiles, unrolled by a factor KWID
			for (int pwi = 0; pwi < WGD; pwi += KWID) {
				
				for (int _pit = 0; _pit < KWID; _pit += 1) {
					int kg = pwi + _pit;

					// Loads data: local --> private (matrix A and B)
					
					for (int _mi = 0; _mi < MWID; _mi += 1) {
						LocalToPrivateDirectA(apd[_mi], alm, _mi, kg, false);
					}
					
					for (int _ni = 0; _ni < NWID; _ni += 1) {
						LocalToPrivateDirectB(bpd[_ni], blm, _ni, kg, true);
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
		for (; kwg < args.patch_size; ++kwg) {

			// Loads data: off-chip --> private (matrix A and B)
			
			for (int _mi = 0; _mi < MWID; _mi += 1) {
				GlobalToPrivateDirectA(apd[_mi], colgms, _mi, args.num_patches, col_offset_batch, idm, kwg, false, false);
			}
			
			for (int _ni = 0; _ni < NWID; _ni += 1) {
				GlobalToPrivateDirectB(bpd[_ni], kernelgms, _ni, args.patch_size, args.kernel_offset, idn, kwg, true, false);
			}

			// Performs the accumulation (Cpmd += Apmd * Bpmd)
			
			for (int _ni = 0; _ni < NWID; _ni += 1) {
				
				for (int _mi = 0; _mi < MWID; _mi += 1) {
					MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);
				}
			}
		}

		// Stores a tile of results
		
		for (int _ni = 0; _ni < NWID; _ni += 1) {
			
			for (int _mi = 0; _mi < MWID; _mi += 1) {
				StoreResultsDirect(resultgm, cpd[_ni * MWID + _mi], _mi, _ni, idm, idn,
													 ONE, ZERO, args.num_patches, result_offset_batch, false);
			}
		}
	}

	// Simple but slower version for the parts on the edge (incomplete tiles in M and N-dimensions)
	else {
		// Loops over all complete workgroup tiles (K-dimension)
		int kwg = 0;
		for (; kwg < (args.patch_size/WGD) * WGD; kwg+=WGD) {

			// Loads data: off-chip --> local
			GlobalToLocalCheckedA(colgms, alm, args.num_patches, col_offset_batch, kwg, false, false, args.num_patches, args.patch_size);
			GlobalToLocalCheckedB(kernelgms, blm, args.patch_size, args.kernel_offset, kwg, true, false, args.num_kernels, args.patch_size);
			barrier();

			// Loops over all workitem tiles, unrolled by a factor KWID
			for (int pwi = 0; pwi < WGD; pwi += KWID) {
				
				for (int _pit = 0; _pit < KWID; _pit += 1) {
					int kg = pwi + _pit;

					// Loads data: local --> private
					
					for (int _mi = 0; _mi < MWID; _mi += 1) {
						LocalToPrivateDirectA(apd[_mi], alm, _mi, kg, false);
					}
					
					for (int _ni = 0; _ni < NWID; _ni += 1) {
						LocalToPrivateDirectB(bpd[_ni], blm, _ni, kg, true);
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
		for (; kwg < args.patch_size; ++kwg) {

			// Loads data: off-chip --> private
			
			for (int _mi = 0; _mi < MWID; _mi += 1) {
				GlobalToPrivateCheckedA(apd[_mi], colgms, _mi, args.num_patches, col_offset_batch, idm, kwg, false, false, args.num_patches);
			}
			
			for (int _ni = 0; _ni < NWID; _ni += 1) {
				GlobalToPrivateCheckedB(bpd[_ni], kernelgms, _ni, args.patch_size, args.kernel_offset, idn, kwg, true, false, args.num_kernels);
			}

			// Performs the accumulation (C += A * B)
			
			for (int _ni = 0; _ni < NWID; _ni += 1) {
				
				for (int _mi = 0; _mi < MWID; _mi += 1) {
					MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);
				}
			}
		}

		// Stores a tile of results
		
		for (int _ni = 0; _ni < NWID; _ni += 1) {
			
			for (int _mi = 0; _mi < MWID; _mi += 1) {
				StoreResultsChecked(resultgm, cpd[_ni * MWID + _mi], _mi, _ni, idm, idn, args.num_patches, args.num_kernels,
														ONE, ZERO, args.num_patches, result_offset_batch, false);
			}
		}
	}
}

// =================================================================================================

// End of the C++11 raw string literal
//)"

// =================================================================================================
