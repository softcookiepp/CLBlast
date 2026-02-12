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

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer inp_buf { real inp[]; };
	layout(binding = 1, std430) writeonly buffer asum_buf { real asum[]; };
#endif

layout(push_constant) uniform XasumEpilogue
{
#if USE_BDA
	const __global real* restrict inp,
	__global real* asum,
#endif
	int asum_offset;
} args;

shared real lm[WGS2];

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

	// Computes the absolute value and stores the final result
	if (lid == 0) {
		#if (PRECISION == 3232 || PRECISION == 6464) && defined(ROUTINE_ASUM)
			asum[args.asum_offset].x = lm[0].x + lm[0].y; // the result is a non-complex number
		#else
			asum[args.asum_offset] = lm[0];
		#endif
	}
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
