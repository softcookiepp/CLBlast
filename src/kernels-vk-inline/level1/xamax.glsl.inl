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

// The main reduction kernel, performing the loading and the majority of the operation
#if RELAX_WORKGROUP_SIZE == 0
	layout(local_size_x = WGS1, local_size_y = 1, local_size_z = 1) in;
#endif

#if USE_BDA == 0
	layout(binding = 0, std430) buffer xgm_buf { real xgm[]; };
	layout(binding = 1, std430) buffer maxgm_buf { singlereal maxgm[]; };
	layout(binding = 2, std430) buffer imaxgm_buf { uint imaxgm[]; };
#endif

layout(push_constant) uniform Xamax
{
	int n;
#if USE_BDA
	__global real* restrict xgm,
#endif
	int x_offset;
	int x_inc;
#if USE_BDA
	__global singlereal* maxgm,
	__global uint* imaxgm
#endif
	int num_groups_0; // because this is not exposed in Vulkan :c
} args;

shared singlereal maxlm[WGS1];
shared uint imaxlm[WGS1];

void main()
{
	const int lid = get_local_id(0);
	const int wgid = get_group_id(0);
	const int num_groups = args.num_groups_0;

	// Performs loading and the first steps of the reduction
	#if defined(ROUTINE_MAX) || defined(ROUTINE_MIN) || defined(ROUTINE_AMIN)
		singlereal max = SMALLEST;
	#else
		singlereal max = ZERO;
	#endif
	uint imax = 0;
	int id = wgid*WGS1 + lid;
	while (id < args.n) {
		const int x_index = id*args.x_inc + args.x_offset;
		#if PRECISION == 3232 || PRECISION == 6464
			precise singlereal x = abs(xgm[x_index].x) + abs(xgm[x_index].y);
		#else
			precise singlereal x = xgm[x_index];
		#endif
		#if defined(ROUTINE_MAX) // non-absolute maximum version
			// nothing special here
		#elif defined(ROUTINE_MIN) // non-absolute minimum version
			x = -x;
		#elif defined(ROUTINE_AMIN) // absolute minimum version
			x = -abs(x);
		#else
			x = abs(x);
		#endif
		precise singlereal dif = x - max;
		if (dif > ZERO) {
			max = x;
			imax = id;
		}
		id += WGS1*num_groups;
	}
	maxlm[lid] = max;
	imaxlm[lid] = imax;
	barrier();

	// Performs reduction in local memory
	for (int s=WGS1/2; s>0; s=s>>1) {
		if (lid < s) {
			if (maxlm[lid + s] > maxlm[lid]) {
				maxlm[lid] = maxlm[lid + s];
				imaxlm[lid] = imaxlm[lid + s];
			}
		}
		barrier();
	}

	// Stores the per-workgroup result
	if (lid == 0) {
		maxgm[wgid] = maxlm[0];
		imaxgm[wgid] = imaxlm[0];
	}
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
