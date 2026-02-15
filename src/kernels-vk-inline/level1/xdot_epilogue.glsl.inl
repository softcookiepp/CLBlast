

// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//	 Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xdot kernel. It implements a dot-product computation using reduction
// kernels. Reduction is split in two parts. In the first (main) kernel the X and Y vectors are
// multiplied, followed by a per-thread and a per-workgroup reduction. The second (epilogue) kernel
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

// The epilogue reduction kernel, performing the final bit of the sum operation. This kernel has to
// be launched with a single workgroup only.
#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = WGS2, local_size_y = 1, local_size_z = 1) in;
#endif

#if USE_BDA == 0
	layout(binding = 0, std430) readonly buffer inp_buffer { real inp[]; };
	layout(binding = 1, std430) writeonly buffer dot_buffer { real dot[]; };
#endif

layout(push_constant) uniform XdotEpilogue
{
#if USE_BDA
	const __global real* restrict inp,
	__global real* dot,
#endif
	int dot_offset;
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

	// Stores the final result
	if (lid == 0) {
		dot[args.dot_offset] = lm[0];
	}
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
