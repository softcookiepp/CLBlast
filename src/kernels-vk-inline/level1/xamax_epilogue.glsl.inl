
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

// literal). Comment-out this line for syntax-highlighting when developing.
R"(
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

#if USE_BDA
	// not yet implemented
#else
	layout(binding = 0, std430) readonly buffer maxgm_buf { singlereal maxgm[]; };
	layout(binding = 1, std430) readonly buffer imaxgm_buf { uint imaxgm[]; };
	layout(binding = 2, std430) writeonly buffer imax_buf { uint imax[]; };
#endif

layout(push_constant) uniform XamaxEpilogue
{
#if USE_BDA
	const __global singlereal* restrict maxgm,
	const __global unsigned int* restrict imaxgm,
	__global unsigned int* imax,
#endif
	int imax_offset;
} args;

shared singlereal maxlm[WGS2];
shared uint imaxlm[WGS2];

void main()
{
	const int lid = get_local_id(0);

	// Performs the first step of the reduction while loading the data
	if (maxgm[lid + WGS2] > maxgm[lid])
	{
		maxlm[lid] = maxgm[lid + WGS2];
		imaxlm[lid] = imaxgm[lid + WGS2];
	}
	else {
		maxlm[lid] = maxgm[lid];
		imaxlm[lid] = imaxgm[lid];
	}
	barrier();

	// Performs reduction in local memory
	for (int s=WGS2/2; s>0; s=s>>1) {
		if (lid < s) {
			if (maxlm[lid + s] > maxlm[lid]) {
				maxlm[lid] = maxlm[lid + s];
				imaxlm[lid] = imaxlm[lid + s];
			}
		}
		barrier();
	}

	// Stores the final result
	if (lid == 0) {
		imax[args.imax_offset] = imaxlm[0];
	}
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
