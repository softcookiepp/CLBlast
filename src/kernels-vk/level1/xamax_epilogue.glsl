
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xamax kernel. It implements index of (absolute) min/max computation using
// reduction kernels. Reduction is split in two parts. In the first (main) kernel the X vector is
// loaded, followed by a per-thread and a per-workgroup reduction. The second (epilogue) kernel
// is executed with a single workgroup only, computing the final result.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
//R"(

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
#if RELAX_WORKGROUP_SIZE == 1
#else
	// how do we want to do this?
	//__kernel __attribute__((reqd_work_group_size(WGS2, 1, 1)))
	DEFINE_LOCAL_SIZE(WGS2, 1, 1)
#endif

#if USE_BDA
	// not yet implemented
#else
	layout(binding = 0, std430) buffer maxgm_buf { singlereal maxgm[]; };
	layout(binding = 1, std430) buffer imaxgm_buf { uint imaxgm[]; };
	layout(binding = 2, std430) buffer imax_buf { uint imax[]; };
#endif

layout(push_constant) push
{
#if USE_BDA
	// not yet implemented, but keeping these here to remember easily
	const __global singlereal* restrict maxgm;
	const __global unsigned int* restrict imaxgm;
	__global unsigned int* imax;
#endif
	const int imax_offset
} args;

shared singlereal maxlm[WGS2];
shared uint imaxlm[WGS2];

// XamaxEpilogue
void main(const __global singlereal* restrict maxgm,
									 const __global unsigned int* restrict imaxgm,
									 __global unsigned int* imax, const int imax_offset) {
	const int lid = gl_LocalInvocationID.x;

	// Performs the first step of the reduction while loading the data
	if (maxgm[lid + WGS2] > maxgm[lid]) {
		maxlm[lid] = maxgm[lid + WGS2];
		imaxlm[lid] = imaxgm[lid + WGS2];
	}
	else {
		maxlm[lid] = maxgm[lid];
		imaxlm[lid] = imaxgm[lid];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// Performs reduction in local memory
	for (int s=WGS2/2; s>0; s=s>>1) {
		if (lid < s) {
			if (maxlm[lid + s] > maxlm[lid]) {
				maxlm[lid] = maxlm[lid + s];
				imaxlm[lid] = imaxlm[lid + s];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Stores the final result
	if (lid == 0) {
		imax[imax_offset] = imaxlm[0];
	}
}

// =================================================================================================

// End of the C++11 raw string literal
//)"

// =================================================================================================
