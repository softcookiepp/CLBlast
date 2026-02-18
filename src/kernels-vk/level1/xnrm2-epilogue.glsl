#version 450
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xnrm2 kernel. It implements a squared norm computation using reduction
// kernels. Reduction is split in two parts. In the first (main) kernel the X vector is squared,
// followed by a per-thread and a per-workgroup reduction. The second (epilogue) kernel
// is executed with a single workgroup only, computing the final result.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
//R"(
#include "../common.glsl"
#include "level1.glsl"
// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library.
#ifndef WGS1
	#define WGS1 64		 // The local work-group size of the main kernel
#endif
#ifndef WGS2
	#define WGS2 64		 // The local work-group size of the epilogue kernel
#endif

// =================================================================================================

// The epilogue reduction kernel, performing the final bit of the operation. This kernel has to
// be launched with a single workgroup only.

#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = WGS2, local_size_y = 1, local_size_z = 1) in;
#endif

#if USE_BDA == 0
	layout(binding = 0, std430) buffer inp_buf { real inp[]; };
	layout(binding = 1, std430) buffer nrm2_buf { real nrm2[]; };
#endif

layout(push_constant) uniform Xnrm2Epilogue
{
#if USE_BDA
	const __global real* restrict inp;
	__global real* nrm2;
#endif
	int nrm2_offset;
};

shared real lm[WGS2];

// Xnrm2Epilogue
void main()
{
	const int lid = get_local_id(0);

	// Performs the first step of the reduction while loading the data
	Add(lm[lid], inp[lid], inp[lid + WGS2]);
	barrier();

	// Performs reduction in local memory
	for (int s=WGS2/2; s>0; s=s>>1) {
		if (lid < s) {
			Add(lm[lid], lm[lid], lm[lid + s]);
		}
		barrier();
	}

	// Computes the square root and stores the final result
	if (lid == 0) {
		#if PRECISION == 3232 || PRECISION == 6464
			nrm2[nrm2_offset].x = sqrt(lm[0].x); // the result is a non-complex number
		#else
			nrm2[nrm2_offset] = sqrt(lm[0]);
		#endif
	}
}

// =================================================================================================

// End of the C++11 raw string literal
//)"

// =================================================================================================
