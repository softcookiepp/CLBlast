#version 450

// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xasum kernel. It implements a absolute sum computation using reduction
// kernels. Reduction is split in two parts. In the first (main) kernel the X vector is loaded,
// followed by a per-thread and a per-workgroup reduction. The second (epilogue) kernel
// is executed with a single workgroup only, computing the final result.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
//R"(
#include "../common.glsl"
// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library.
#ifndef LEVEL1_USE_SPEC_CONSTANTS
	#define LEVEL1_USE_SPEC_CONSTANTS 0
#endif

#if LEVEL1_USE_SPEC_CONSTANTS == 1
	#ifdef WGS1
		#undef WGS1
	#endif
	layout(constant_id = 0) const int WGS1 = 64; // The local work-group size of the main kernel
	
	#ifdef WGS2
		#undef WGS2
	#endif
	//layout(constant_id = 1) const int WGS2 = 64; // The local work-group size of the epilogue kernel
	
	#ifdef ROUTINE_SUM
		#undef ROUTINE_SUM
	#endif
	layout(constant_id = 1) const int ROUTINE_SUM = 0;
#else
	#ifndef WGS1
		#define WGS1 64		 // The local work-group size of the main kernel
	#endif
	#ifndef WGS2
		#define WGS2 64		 // The local work-group size of the epilogue kernel
	#endif
	
	#ifndef ROUTINE_SUM
		#define ROUTINE_SUM 0
	#endif
#endif



// =================================================================================================

// The main reduction kernel, performing the loading and the majority of the operation
#if LEVEL1_USE_SPEC_CONSTANTS == 1
	layout(local_size_x_id = 0) in;
#else
	layout(local_size_x = WGS1, local_size_y = 1, local_size_z = 1) in;
#endif

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer xgm_buf { real xgm[]; };
	layout(binding = 1, std430) writeonly buffer outp_buf { real outp[]; };
#endif

layout(push_constant) uniform Xasum
{
	uint n;
	#if USE_BDA
		real_ptr_t xgm;
	#endif
	uint x_offset;
	uint x_inc;
	#if USE_BDA
		real_ptr_t outp;
	#endif
};

shared real lm[WGS1];

void main()
{
	const uint lid = gl_LocalInvocationID[0];
	const uint wgid = gl_WorkGroupID[0];
	const uint num_groups = gl_NumWorkGroups.x;

	// Performs loading and the first steps of the reduction
	real acc;
	SetToZero(acc);
	uint id = wgid*WGS1 + lid;
	while (id < n) {
		real x = indexGM(xgm, id*x_inc + x_offset);
		if (ROUTINE_SUM == 0) // if not the sum routine, take the absolute value
			AbsoluteValue(x);
		Add(acc, acc, x);
		id += WGS1*num_groups;
	}
	lm[lid] = acc;
	barrier();

	// Performs reduction in local memory
	for (uint s=WGS1/2; s>0; s=s>>1) {
		if (lid < s) {
			Add(lm[lid], lm[lid], lm[lid + s]);
		}
		barrier();
	}

	// Stores the per-workgroup result
	if (lid == 0) {
		indexGM(outp, wgid) = lm[0];
	}
}

// =================================================================================================

// End of the C++11 raw string literal
//)"

// =================================================================================================
